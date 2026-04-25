#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image                 #
from torchvision import transforms    #
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


def get_perturbed_cam(viewpoint_cam, translation_std=0.002, rotation_std=0.001):
    """
    Slightly jiggles the camera extrinsics to prevent billboard overfitting.
    """
    import copy
    # Create a shallow copy to avoid modifying the original dataset cameras
    p_cam = copy.copy(viewpoint_cam)
    
    # Random translation shift
    shift = torch.randn(3, device="cuda") * translation_std
    
    # Apply to the world-to-view matrix
    # p_cam.world_view_transform is [4, 4]
    p_cam.world_view_transform = viewpoint_cam.world_view_transform.clone()
    p_cam.world_view_transform[3, :3] += shift
    
    # Recompute the full projection transform used by the rasterizer
    p_cam.full_proj_transform = (p_cam.world_view_transform @ p_cam.projection_matrix)
    p_cam.camera_center = p_cam.world_view_transform.inverse()[3, :3]
    
    return p_cam

from PIL import Image
from torchvision import transforms

def load_lr_gt(viewpoint_cam, dataset_path):                                    ####
    # Split at the first dot and take only the prefix (e.g., 'image006')
    base_name = viewpoint_cam.image_name.split('.')[0]
    
    lr_dir = os.path.join(dataset_path, "images_lr")
    
    # We check for both common extensions to find the actual file
    for ext in [".jpg", ".png", ".JPG", ".PNG"]:
        potential_path = os.path.join(lr_dir, f"{base_name}{ext}")
        if os.path.exists(potential_path):
            lr_img = Image.open(potential_path).convert("RGB")
            return transforms.ToTensor()(lr_img).to("cuda")
            
    # If we get here, the file truly doesn't exist in that folder
    raise FileNotFoundError(f"Could not find any LR image for '{base_name}' in {lr_dir}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    # Around line 40 in train.py, after "scene = Scene(...)"
    print(f"DEBUG: Camera 0 resolution is {scene.getTrainCameras()[0].image_width}x{scene.getTrainCameras()[0].image_height}")
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0


    # --- ADD THIS: Initialize S2Gaussian Tracking ---
    flag_grads = {}
    s2_eps = 0.1 # The epsilon decay factor from the paper
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if iteration < opt.iterations:
            render_cam = get_perturbed_cam(viewpoint_cam, translation_std=0.001) 
        else:
            render_cam = viewpoint_cam

# Render with the jiggled camera
        render_pkg = render(render_cam, gaussians, pipe, bg, 
                    use_trained_exp=dataset.train_test_exp, 
                    separate_sh=SPARSE_ADAM_AVAILABLE)
        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        # gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # if FUSED_SSIM_AVAILABLE:
        #     ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        # else:
        #     ssim_value = ssim(image, gt_image)

        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)




        ######################################################################################################
        
        gt_image_hr = viewpoint_cam.original_image.cuda()
        
        # 2. Get the actual LR ground truth (the original un-processed images)
        gt_image_lr = load_lr_gt(viewpoint_cam, dataset.source_path)
        
        # 3. Render at HR (this is what the 'render' function already does)
        image_hr = render_pkg["render"]
        
        # 4. Downsample the rendered HR image to match LR resolution
        # We use area interpolation as it's standard for downsampling
        lr_height, lr_width = gt_image_lr.shape[1], gt_image_lr.shape[2]
        image_lr_rendered = torch.nn.functional.interpolate(
            image_hr.unsqueeze(0), 
            size=(lr_height, lr_width), 
            mode='area'
        ).squeeze(0)

        # 5. Compute HR Loss (vs. Super-resolved images)
        Ll1_hr = l1_loss(image_hr, gt_image_hr)
        # Use fused_ssim if available for speed, otherwise standard ssim
        if FUSED_SSIM_AVAILABLE:
            ssim_hr = fused_ssim(image_hr.unsqueeze(0), gt_image_hr.unsqueeze(0))
        else:
            ssim_hr = ssim(image_hr, gt_image_hr)
        
        # Standard 3DGS formula: (1-lambda)*L1 + lambda*(1-SSIM)
        loss_hr = (1.0 - opt.lambda_dssim) * Ll1_hr + opt.lambda_dssim * (1.0 - ssim_hr)
        
        # 6. Compute LR Loss (Anchor Loss)
        Ll1_lr = l1_loss(image_lr_rendered, gt_image_lr)
        if FUSED_SSIM_AVAILABLE:
            ssim_lr = fused_ssim(image_lr_rendered.unsqueeze(0), gt_image_lr.unsqueeze(0))
        else:
            ssim_lr = ssim(image_lr_rendered, gt_image_lr)
        
        loss_lr = (1.0 - opt.lambda_dssim) * Ll1_lr + opt.lambda_dssim * (1.0 - ssim_lr)
        
        # 7. Total Weighted Loss (using your existing scheduling)
        loss = (curr_lambda_hr * loss_hr) + (curr_lambda_lr * loss_lr)


        ######################################################################################################
        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            for group in gaussians.optimizer.param_groups:
                name = group['name']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    curr_grad = p.grad.data
                    num_gaussians = curr_grad.shape[0]
                    
                    # Handle Densification/Pruning shape changes by resetting the flags
                    # (In standard 3DGS, the number of Gaussians changes dynamically)
                    if name not in flag_grads or flag_grads[name].shape != curr_grad.shape:
                        flag_grads[name] = torch.zeros_like(curr_grad)
                    
                    flag = flag_grads[name]
                    
                    # Flatten the gradient per-Gaussian to compute alignment
                    # (e.g., handles 3D positions, 1D opacities, and ND SH features)
                    curr_grad_flat = curr_grad.view(num_gaussians, -1)
                    flag_flat = flag.view(num_gaussians, -1)
                    
                    # Dot product > 0 is mathematically equivalent to Cosine Similarity > 0
                    dot_product = (curr_grad_flat * flag_flat).sum(dim=-1)
                    
                    # Reshape the boolean mask so it broadcasts over the original parameter shape
                    mask_positive = (dot_product > 0).view(num_gaussians, *([1] * (curr_grad.dim() - 1)))
                    
                    # Apply S2Gaussian conditional logic
                    # If aligned: keep grad, update flag to average
                    # If divergent: dampen grad by eps, update flag with momentum
                    new_curr_grad = torch.where(mask_positive, curr_grad, s2_eps * curr_grad)
                    new_flag = torch.where(mask_positive, (flag + curr_grad) / 2.0, (1.0 - s2_eps) * flag + s2_eps * curr_grad)
                    
                    # Overwrite the gradients for the optimizer to use
                    p.grad.data = new_curr_grad
                    
                    # Save the updated flag gradient for the next iteration
                    flag_grads[name] = new_flag
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1_hr, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
