import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, weighted_ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
from torchvision import transforms

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


import torch.nn.functional as F

def generate_dynamic_weight_map(hr_image, lr_image):
    # hr_image: [3, H, W], lr_image: [3, h, w]
    # 1. Upsample LR to HR size
    lr_upsampled = F.interpolate(lr_image.unsqueeze(0), size=(hr_image.shape[1], hr_image.shape[2]), mode='bicubic').squeeze(0)
    
    # 2. Compute absolute difference (grayscale)
    diff = torch.abs(hr_image - lr_upsampled).mean(dim=0)
    
    # 3. Normalize between 0.1 and 1.0 (so no pixel has 0 weight)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5)
    weight_map = 0.1 + 0.9 * diff
    
    return weight_map

def load_lr_gt(viewpoint_cam, dataset_path):
    # Logic from Script 1: Find LR ground truth in 'images_lr' folder
    base_name = viewpoint_cam.image_name.split('.')[0]
    lr_dir = os.path.join(dataset_path, "images_lr")
    for ext in [".jpg", ".png", ".JPG", ".PNG"]:
        potential_path = os.path.join(lr_dir, f"{base_name}{ext}")
        if os.path.exists(potential_path):
            lr_img = Image.open(potential_path).convert("RGB")
            return transforms.ToTensor()(lr_img).to("cuda")
    raise FileNotFoundError(f"Could not find any LR image for '{base_name}' in {lr_dir}")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed.")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Load Weight Maps (Logic from Script 2)
    for cam in scene.getTrainCameras():
        if args.weight_maps_path is not None:
            sr_weight_map_lr = torch.load(os.path.join(args.weight_maps_path, "SR_" + cam.image_name + ".pty")).cuda()
            cam.sr_weight_map = torch.nn.functional.interpolate(sr_weight_map_lr.unsqueeze(0).unsqueeze(0), (cam.image_height, cam.image_width))[0][0]
        else:
           gt_hr = cam.original_image.cuda()
           gt_lr = load_lr_gt(cam, dataset.source_path)
           cam.sr_weight_map = generate_dynamic_weight_map(gt_hr, gt_lr)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer)["render"]
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image_hr = render_pkg["render"]

        # --- HR/LR Integrated Loss ---
        gt_image_hr = viewpoint_cam.original_image.cuda()
        gt_image_lr = load_lr_gt(viewpoint_cam, dataset.source_path) # Dynamic loading like Script 1

        # Downsample rendered image to LR resolution
        lr_h, lr_w = gt_image_lr.shape[1], gt_image_lr.shape[2]
        image_lr_rendered = torch.nn.functional.interpolate(image_hr.unsqueeze(0), size=(lr_h, lr_w), mode='area').squeeze(0)

        # Weighted HR Loss using sr_weight_map from Script 2
        Ll1_hr = (torch.abs(image_hr - gt_image_hr) * viewpoint_cam.sr_weight_map.unsqueeze(0)).mean()
        ssim_hr = weighted_ssim(image_hr, gt_image_hr, viewpoint_cam.sr_weight_map)
        loss_hr = (1.0 - opt.lambda_dssim) * Ll1_hr + opt.lambda_dssim * (1.0 - ssim_hr)

        # LR Loss
        Ll1_lr = l1_loss(image_lr_rendered, gt_image_lr)
        ssim_lr = ssim(image_lr_rendered, gt_image_lr)
        loss_lr = (1.0 - opt.lambda_dssim) * Ll1_lr + opt.lambda_dssim * (1.0 - ssim_lr)

        loss = (args.gamma * loss_hr) + ((1 - args.gamma) * loss_lr)

        # Depth regularization
        Ll1depth = 0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)

            # --- Reporting from Script 1 ---
            training_report(tb_writer, iteration, Ll1_hr, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            if iteration in saving_iterations:
                scene.save(iteration)

            # Densification & Optimizer Steps
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(gaussians.max_radii2D[render_pkg["visibility_filter"]], render_pkg["radii"][render_pkg["visibility_filter"]])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], render_pkg["visibility_filter"])
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, None, render_pkg["radii"])
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss/l1_hr', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss/total', loss.item(), iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras']:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                print(f"\n[ITER {iteration}] {config['name']} PSNR: {psnr_test/len(config['cameras'])}")
        torch.cuda.empty_cache()

def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    os.makedirs(args.model_path, exist_ok=True)
    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--weight_maps_path", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.4)
    args = parser.parse_args(sys.argv[1:])
    
    safe_state(False)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
