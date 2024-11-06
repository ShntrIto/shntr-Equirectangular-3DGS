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
from utils.loss_utils import l1_loss, ssim, latitude_weight
from gaussian_renderer import render, render_spherical, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import math
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, simple_mask, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, panorama, device):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, panorama=panorama)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    mask_H = scene.getTrainCameras()[0].image_height
    mask_W = scene.getTrainCameras()[0].image_width
    image_mask = torch.ones((3, mask_H, mask_W), device=device)
    simple_mask = simple_mask[0]
    if simple_mask is not None:
        # image_mask[:, :int(mask_H*simple_mask), :] = 0.0
        image_mask[:, int(mask_H*(1-simple_mask)):, :] = 0.0
        image_mask = image_mask.bool()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
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
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
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
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # weights = latitude_weight(viewpoint_cam.image_height).to(device)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device=device) if opt.random_background else background
        if viewpoint_cam.panorama:
            render_pkg = render_spherical(viewpoint_cam, gaussians, pipe, bg)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        
        # viewspace_points は，画像座標系におけるガウシアンの位置を表す（2次元座標）
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # import matplotlib.pyplot as plt
        # if depth is not None:
        #     plt.imshow(depth.squeeze().detach().cpu().numpy(), cmap='viridis')
        #     plt.savefig('depth.png')
        #     plt.close()
        # if iteration % 2000 == 0:
        #     import numpy as np
        #     depth = render_pkg["rendered_depth"] if "rendered_depth" in render_pkg else None # depth
        #     np.save('depth.npy', depth.squeeze().detach().cpu().numpy())
        #     import pdb; pdb.set_trace()
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.panorama:
            if simple_mask is None:
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            else:
                masked_image = image[image_mask].view(image.shape[0], -1, image.shape[-1]) 
                masked_gt = gt_image[image_mask].view(image.shape[0], -1, image.shape[-1])
                Ll1 = l1_loss(masked_image, masked_gt)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(masked_image, masked_gt))
                # Ll1 = l1_loss(image, gt_image, weights=image_mask)
                # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, weights=weights))
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, render_spherical, (pipe, background), device)
            if panorama:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_spherical, (pipe, background), device)
            else:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), device)
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
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc,  renderArgs, device):
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
                    depth = renderFunc(viewpoint, scene.gaussians, *renderArgs)["plane_depth"]
                    normal = renderFunc(viewpoint, scene.gaussians, *renderArgs)["rendered_normal"]
                    # normal = renderFunc(viewpoint, scene.gaussians, *renderArgs)["rendered_normal"]
                    gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
    
                        import numpy as np
                        import matplotlib.pyplot as plt
                        depth = (depth - depth.min()) / (depth.max() - depth.min()) # デプスマップを正規化
                        depth_np = depth.cpu().numpy()
                        depth_colored = plt.get_cmap('turbo')(depth_np)[:, :, :, :3] # alpha を除く
                        depth_colored = torch.from_numpy(depth_colored).permute(0, 3, 1, 2) # CHW に変換
                        
                        normal = normal.permute(1,2,0)
                        normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
                        normal = normal.cpu().numpy()
                        normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)
                        normal = torch.from_numpy(normal).permute(2, 0, 1)[None, ...] # CHW に変換
                        
                        # confidence_np = 1 - confidence.cpu().numpy() # 信頼度を反転
                        # confidence_colored = plt.get_cmap('Reds')(confidence_np)[:, :, :, :3] # alpha を除く
                        # confidence_colored = torch.from_numpy(confidence_colored).permute(0, 3, 1, 2) # CHW に変換
    
                        tb_writer.add_images(config['name'] + "_view_{}/render_plane_depth".format(viewpoint.image_name), depth_colored, global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render_normal".format(viewpoint.image_name), normal, global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/render_conf".format(viewpoint.image_name), confidence_colored, global_step=iteration)
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--panorama", action="store_true")
    parser.add_argument("--simple_mask", nargs="+", type=float, default=None)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("-d", "--device_num", type=str, default="0")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    device = "cuda:" + args.device_num
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.simple_mask, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.panorama, device)

    # All done
    print("\nTraining complete.")