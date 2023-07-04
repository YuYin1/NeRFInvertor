import argparse
from curses import meta
from dis import dis
from itertools import cycle
from locale import normalize
import os
import sys
from random import triangular
from sqlite3 import PARSE_DECLTYPES
from textwrap import indent
from turtle import pos

from sklearn.datasets import load_diabetes
from sklearn.metrics import zero_one_loss
from grpc import metadata_call_credentials
import numpy as np
import math
from collections import deque
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import torchvision.transforms as transforms
import importlib
import time
# import trimesh
# from discriminators import discriminators
# from siren import siren
from generators import generators
import configs
# import fid_evaluation
import datasets
from tqdm import tqdm
from datetime import datetime
import copy
from torch_ema import ExponentialMovingAverage
# import pytorch3d
# from loss import *
from torch.utils.tensorboard import SummaryWriter
import pickle, PIL
from PIL import Image
# import utils
import dnnlib
from utils.arcface import get_model

torch.backends.cudnn.benchmark = True


# sample noises
def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
        # torch.randn - sample random numbers from a normal distribution with mean 0 and varaiance 1
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
        # torch.rand - sample random numbers froma uniform distribution
    return z

##### --------------------------------------- set the networks ---------------------------------------------------


def load_models_for_loss(device, opt):
    #for LPIPS loss
    if opt.config.find('FACES') >= 0:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)
    elif opt.config.find('CATS') >= 0:   # CATS, CARLA
        import lpips
        vgg16 = lpips.LPIPS(net='vgg').eval().to(device) # closer to "traditional" perceptual loss, when used for optimization
    print("load vgg for LPIPS loss")

    face_recog = get_model('r50', fp16=False)
    face_recog.load_state_dict(torch.load('pretrained_models/arcface.pth'))
    print("load face_recog model for ID loss")
    id_loss = IDLoss(face_recog.eval()).to(device)

    return vgg16, id_loss 


# define generator
def set_generator(config, device, opt):
    generator_args = {}
    if 'representation' in config['generator']:
        generator_args['representation_kwargs'] = config['generator']['representation']['kwargs']
    if 'renderer' in config['generator']:
        generator_args['renderer_kwargs'] = config['generator']['renderer']['kwargs']
    generator = getattr(generators, config['generator']['class'])(
        **generator_args,
        **config['generator']['kwargs']
    )

    print(f"Loaded pretrained network: {opt.pretrain_model}")
    if opt.pretrain_model != '':
        generator.load_state_dict(torch.load(opt.pretrain_model, map_location='cpu'))

    generator = generator.to(device)

    if opt.pretrain_model != '':
        print(f"loaded ema network!")
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

        ema = torch.load(opt.pretrain_model.replace('generator.pth','ema.pth'), map_location=device)
        parameters = [p for p in generator.parameters() if p.requires_grad]
        ema.copy_to(parameters)        
    else:
        # exponential moving avaerage is to place a greater weight on the most recent data points
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    return generator, ema, ema2


# set IDLoss
class IDLoss(nn.Module):
    def __init__(self, facenet):
        super(IDLoss, self).__init__()
        self.facenet = facenet

    def forward(self, x, y):
        x = F.interpolate(x, size=[112, 112], mode='bilinear')
        y = F.interpolate(y, size=[112, 112], mode='bilinear')

        # x = 2*(x-0.5)
        # y = 2*(y-0.5)
        feat_x = self.facenet(x)
        feat_y = self.facenet(y.detach())

        loss = 1 - F.cosine_similarity(feat_x, feat_y, dim=-1)

        return loss

##### ------------------------------------------- set the optimizers ---------------------------------------------------

# define optimizer
def set_optimizer_G(generator_ddp, config, opt):
    param_groups = []
    if 'mapping_network_lr' in config['optimizer']:
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if 'module.representation.rf_network.mapping_network' in n]
        param_groups.append({'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':config['optimizer']['mapping_network_lr']})
    if 'sampling_network_lr' in config['optimizer']:
        sampling_network_parameters = [p for n, p in generator_ddp.named_parameters() if 'module.representation.sample_network' in n]
        param_groups.append({'params': sampling_network_parameters, 'name': 'sampling_network', 'lr':config['optimizer']['sampling_network_lr']})
    generator_parameters = [p for n, p in generator_ddp.named_parameters() if 
        ('mapping_network_lr' not in config['optimizer'] or 'module.representation.rf_network.mapping_network' not in n) and
        ('sampling_network_lr' not in config['optimizer'] or 'module.representation.sample_network' not in n)]
    param_groups.append({'params': generator_parameters, 'name': 'generator'})
    
    optimizer_G = torch.optim.Adam(param_groups, lr=config['optimizer']['gen_lr'], betas=config['optimizer']['betas'])

    return optimizer_G


def training_step_G(sample_z, sample_pose, input_imgs, zs, real_poses, generator_ddp, ema, ema2, 
    generator_ori_ddp, vgg16, id_loss, optimizer_G, scaler, config, opt, device):
    batch_split = 1
    if opt.load_mask:
        real_imgs = input_imgs[:, :3, :, :]
        mat_imgs = input_imgs[:, 3:, :, :]
    else:
        real_imgs = input_imgs
    bs = zs.size()[0]
    split_batch_size = zs.shape[0] // batch_split  # minibatch split for memory reduction
    img_size = input_imgs.size(-1)

    # --------------------------- interpolate zs and sampled z ---------------------------------
    interpolation_direction = sample_z - zs
    interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
    result_zs = zs + opt.regulizer_alpha * interpolation_direction / interpolation_direction_norm

    gen_imgs_list = []
    losses_dict = {}
    for split in range(batch_split):
        g_loss = 0
        with torch.cuda.amp.autocast():
            subset_z = zs[split * split_batch_size:(split+1) * split_batch_size]
            generator_ddp.module.get_avg_w()
            gen_imgs, gen_bef_aggr, _ = generator_ddp(subset_z, **config['camera'], detailed_output=True, truncation_psi=opt.psi)

            # --------------------------- loss constraint-----------------------            
            if opt.lambda_id > 0:
                id_l = id_loss(gen_imgs, real_imgs).mean() * opt.lambda_id
                g_loss += id_l
                losses_dict['id_l'] = id_l
            if opt.load_mask and opt.lambda_bg_sigma > 0:
                ## force bg sigma to be 0
                rgb_sigma = gen_bef_aggr['outputs']
                N_steps = rgb_sigma.size(-2)
                mat_imgs = mat_imgs.permute(0, 2, 3, 1).expand(-1, -1, -1, N_steps).reshape(bs, -1, N_steps)

                weights = gen_bef_aggr['weights'].reshape(rgb_sigma.size(0), img_size*img_size, N_steps, 1)
                bg_sigma = (1 - mat_imgs[:, :, -1]) * weights[:, :, -1, 0]
                l2_bg_sigma = torch.mean((bg_sigma - 1) ** 2) * opt.lambda_bg_sigma
                
                # # error 4:
                # bg_sigma = (1 - mat_imgs) * weights[:, :, :, 0]
                # l2_bg_sigma = torch.mean(bg_sigma ** 2) * opt.lambda_bg_sigma
                # # error 3:
                # bg_sigma = (1 - mat_imgs) * weights[:, :, :, 0]
                # l2_bg_sigma = torch.mean((bg_sigma - 1) ** 2) * opt.lambda_bg_sigma

                # bg_sigma = (1 - mat_imgs[:, :, -1]) * weights[:, :, -1, 0]
                # l2_bg_sigma = torch.mean((bg_sigma - 1) ** 2) * opt.lambda_bg_sigma
                g_loss += l2_bg_sigma
                losses_dict['l2_bg_sigma'] = l2_bg_sigma

                # gen_imgs = mat_imgs * gen_imgs
                # real_imgs = mat_imgs * real_imgs

                # gen_imgs_bg = (1 - mat_imgs) * gen_imgs
                # real_imgs_bg = (1 - mat_imgs) * real_imgs
                # if opt.lambda_l2 > 0:
                #     l2_bg = torch.mean((gen_imgs_bg - real_imgs_bg) ** 2) * opt.lambda_l2 * 0.1
                #     g_loss += l2_bg
                #     losses_dict['l2_bg'] = l2_bg
                # if opt.lambda_perceptual > 0:
                #     gen_features_bg = vgg16(127.5 * (gen_imgs_bg + 1), resize_images=False, return_lpips=True)
                #     real_features_bg = vgg16(127.5 * (real_imgs_bg + 1), resize_images=False, return_lpips=True)
                #     perceptual_loss_bg = ((1000 * gen_features_bg - 1000 * real_features_bg) ** 2).mean() * opt.lambda_perceptual * 0.1
                #     g_loss += perceptual_loss_bg
                #     losses_dict['perceptual_loss_bg'] = perceptual_loss_bg
            if opt.lambda_l2 > 0:
                l2 = torch.mean((gen_imgs - real_imgs) ** 2) * opt.lambda_l2
                ## l2 = nn.MSELoss()(gen_imgs, real_imgs) * opt.lambda_l2

                # img_size = real_imgs.size(-1)
                # gen_imgs_d2 = F.upsample(gen_imgs, size=(img_size//2,img_size//2), mode='bilinear')
                # real_imgs_d2 = F.upsample(real_imgs, size=(img_size//2,img_size//2), mode='bilinear')
                # l2 += torch.mean((gen_imgs_d2 - real_imgs_d2)**2) * opt.lambda_l2

                # gen_imgs_d4 = F.upsample(gen_imgs, size=(img_size//4,img_size//4), mode='bilinear')
                # real_imgs_d4 = F.upsample(real_imgs, size=(img_size//4,img_size//4), mode='bilinear')
                # l2 += torch.mean((gen_imgs_d4-real_imgs_d4)**2) * opt.lambda_l2
                # l2 = l2 / 3.0

                g_loss += l2
                losses_dict['l2'] = l2
            if opt.lambda_perceptual > 0:
                if opt.config.find('FACES') >= 0:
                    gen_features = vgg16(127.5 * (gen_imgs + 1), resize_images=False, return_lpips=True)
                    real_features = vgg16(127.5 * (real_imgs + 1), resize_images=False, return_lpips=True)
                    perceptual_loss = ((1000 * gen_features - 1000 * real_features) ** 2).mean() * opt.lambda_perceptual

                    # gen_features_d2 = vgg16(127.5*(gen_imgs_d2+1), resize_images=False, return_lpips=True)
                    # real_features_d2 = vgg16(127.5*(real_imgs_d2+1), resize_images=False, return_lpips=True)
                    # perceptual_loss += ((1000*gen_features_d2-1000*real_features_d2)**2).mean() * opt.lambda_perceptual
                    
                    # gen_features_d4 = vgg16(127.5*(gen_imgs_d4+1), resize_images=False, return_lpips=True)
                    # real_features_d4 = vgg16(127.5*(real_imgs_d4+1), resize_images=False, return_lpips=True)
                    # perceptual_loss += ((1000*gen_features_d4-1000*real_features_d4)**2).mean() * opt.lambda_perceptual

                    # perceptual_loss = perceptual_loss / 3.0

                elif opt.config.find('CATS') >= 0:   # CATS, CARLA
                    perceptual_loss = vgg16(gen_imgs, real_imgs).mean() * opt.lambda_perceptual
                    # perceptual_loss += vgg16(gen_imgs_d2, real_imgs_d2).mean() * opt.lambda_perceptual
                    # perceptual_loss += vgg16(gen_imgs_d4, real_imgs_d4).mean() * opt.lambda_perceptual
                    # perceptual_loss = perceptual_loss / 3.0

                g_loss += perceptual_loss
                losses_dict['perceptual_loss'] = perceptual_loss

            # --------------------------- loc_regularization -----------------------
            ## ori G
            subset_sample_z = result_zs[split * split_batch_size:(split + 1) * split_batch_size]
            with torch.no_grad():
                generator_ori_ddp.module.get_avg_w()
                sampled_img, sampled_details, gen_positions = generator_ori_ddp(subset_sample_z, **config['camera'], img_size=128, detailed_output=True, truncation_psi=opt.psi)
            ## finetuned G
            output_updated, details_updated, _ = generator_ddp(subset_sample_z, **config['camera'], img_size=128, camera_pos=gen_positions, detailed_output=True, truncation_psi=opt.psi)


            if opt.lambda_reg_rgbBefAggregation > 0:
                # [0]: pixels, [1]: depth, [2]: weights, [3]: T, [4]: rgb_sigma, [5]: z_vals, [6]: is_valid
                sampled_rgb_bef_aggregation = sampled_details['weights'] * sampled_details['outputs'][..., :3]
                output_rgb_bef_aggregation = details_updated['weights'] * details_updated['outputs'][..., :3]
                reg_rgbBefAggregation = torch.nn.L1Loss()(sampled_rgb_bef_aggregation, output_rgb_bef_aggregation) \
                                    * opt.lambda_reg_rgbBefAggregation
                g_loss += reg_rgbBefAggregation
                losses_dict['reg_rgbBefAggregation'] = reg_rgbBefAggregation
            if opt.lambda_reg_sigmaBefAggregation > 0:
                sampled_sigma_bef_aggregation = sampled_details['outputs'][..., 3:]
                output_sigma_bef_aggregation = details_updated['outputs'][..., 3:]
                reg_sigmaBefAggregation = torch.nn.L1Loss()(sampled_sigma_bef_aggregation, output_sigma_bef_aggregation) \
                                    * opt.lambda_reg_sigmaBefAggregation
                g_loss += reg_sigmaBefAggregation
                losses_dict['reg_sigmaBefAggregation'] = reg_sigmaBefAggregation
            if opt.lambda_reg_volumeDensity > 0:
                reg_volumeDensity = torch.nn.L1Loss()(details_updated['depth'], sampled_details['depth']) * opt.lambda_reg_volumeDensity
                g_loss += reg_volumeDensity
                losses_dict['reg_volumeDensity'] = reg_volumeDensity
            if opt.lambda_loc_reg_l2 > 0:
                reg_l2 = torch.mean((output_updated - sampled_img) ** 2) * opt.lambda_loc_reg_l2
                g_loss += reg_l2
                losses_dict['reg_l2'] = reg_l2
            if opt.lambda_loc_reg_perceptual > 0:
                if opt.config.find('FACES') >= 0:
                    gen_features = vgg16(127.5 * (output_updated + 1), resize_images=False, return_lpips=True)
                    real_features = vgg16(127.5 * (sampled_img + 1), resize_images=False, return_lpips=True)
                    reg_perceptual_loss = ((1000 * gen_features - 1000 * real_features) ** 2).mean() * opt.lambda_perceptual
                elif opt.config.find('CATS') >= 0:   # CATS, CARLA
                    reg_perceptual_loss = vgg16(output_updated, sampled_img).mean() * opt.lambda_perceptual
                g_loss += reg_perceptual_loss
                losses_dict['reg_perceptual_loss'] = reg_perceptual_loss
            gen_imgs_list.append(output_updated)
            gen_imgs_list.append(sampled_img)
            gen_imgs_list.append(gen_imgs)            
        scaler.scale(g_loss).backward()

    scaler.unscale_(optimizer_G)
    torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), config['optimizer'].get('grad_clip', 0.3))
    scaler.step(optimizer_G)
    scaler.update()
    optimizer_G.zero_grad()
    ema.update(generator_ddp.parameters())
    ema2.update(generator_ddp.parameters())

    loss_list = [
        l2.detach() if opt.lambda_l2 else 0,
        perceptual_loss.detach() if opt.lambda_perceptual else 0,
        id_l.detach() if opt.lambda_id else 0,
        reg_rgbBefAggregation.detach() if opt.lambda_reg_rgbBefAggregation else 0,
        reg_sigmaBefAggregation.detach() if opt.lambda_reg_sigmaBefAggregation else 0,
        reg_volumeDensity.detach() if opt.lambda_reg_volumeDensity else 0,
        reg_l2.detach() if opt.lambda_loc_reg_l2 else 0,
        reg_perceptual_loss.detach() if opt.lambda_loc_reg_perceptual else 0,
    ]

    return g_loss.detach(), losses_dict, gen_imgs_list


def training_process(rank, world_size, opt, device):
    # --------------------------------------------------------------------------------------
    # extract training config
    config = getattr(configs, opt.config)
    if rank == 0:
        # print(metadata)
        log_dir = opt.output_dir + '/tensorboard/'
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir, 0)

    # --------------------------------------------------------------------------------------
    # set amp gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    if config['global'].get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)


    # --------------------------------------------------------------------------------------
    # set LPIPS loss and id loss
    vgg16, id_loss = load_models_for_loss(device, opt)

    # --------------------------------------------------------------------------------------
    # set the GRAM generator 
    generator, ema, ema2 = set_generator(config, device, opt)
    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    generator = generator_ddp.module
    generator.renderer.lock_view_dependence = True

    if rank == 0:
        total_num = sum(p.numel() for p in generator_ddp.parameters())
        trainable_num = sum(p.numel() for p in generator_ddp.parameters() if p.requires_grad)
        print('G: Total ', total_num, ' Trainable ', trainable_num)        

    generator_ori, _, _ = set_generator(config, device, opt)
    generator_ori_ddp = DDP(generator_ori, device_ids=[rank], find_unused_parameters=True)
    generator_ori = generator_ori_ddp.module
    generator_ori.eval()

    # --------------------------------------------------------------------------------------
    # set optimizers
    optimizer_G = set_optimizer_G(generator_ddp, config, opt)
    torch.cuda.empty_cache()
    generator_losses = []

    # ----------
    #  Training
    # ----------
    if rank == 0:
        log_file = os.path.join(opt.output_dir, 'logs.txt')
        with open(log_file, 'w') as f:
            f.write(str(opt))
            f.write('\n\n')
            f.write(str(config))
            f.write('\n\n')
            f.write(str(generator))
            f.write('\n\n')


    total_progress_bar = tqdm(total=opt.n_epochs, desc="Total progress", dynamic_ncols=True, disable=True)
    torch.manual_seed(3)

    #--------------------------------------------------------------------------------------
    # get dataset
    dataset = getattr(datasets, config['dataset']['class'])(opt, **config['dataset']['kwargs'])
    dataloader, CHANNELS = datasets.get_dataset_distributed_(
        dataset,
        world_size,
        rank,
        config['global']['batch_size']
    )

    # --------------------------------------------------------------------------------------
    # main training loop
    generator_ddp.train()
    print("Total num epochs = ", opt.n_epochs)
    start_time = time.time()
    for epoch in range(opt.n_epochs):
        total_progress_bar.update(1)
        generator.epoch += 1
        # --------------------------------------------------------------------------------------
        # trainging iterations
        for i, (imgs, poses, zs) in enumerate(dataloader):
            generator.step += 1
            zs = zs.to(device)
            fixed_z = zs

            real_imgs = imgs.to(device, non_blocking=True)
            real_poses = poses.to(device, non_blocking=True)
            generator.v_mean = poses[0, 0]
            generator.h_mean = poses[0, 1]
            generator.h_stddev = generator.v_stddev = 0

            if scaler.get_scale() < 1:
                scaler.update(1.)
            # TRAIN GENERATOR
            ## ------------------------ sample latend codes for regularization ------------------------------
            # sample z
            sample_z = z_sampler((1, 256), device=device, dist='gaussian')
            # sample pose
            yaw = torch.randn((1, 1), device=device) * 0.3 + math.pi * 0.5
            pitch = torch.randn((1, 1), device=device) * 0.155 + math.pi * 0.5
            yaw = torch.clamp(yaw, math.pi * 0.5 - 1.3, math.pi * 0.5 + 1.3)
            pitch = torch.clamp(pitch, math.pi * 0.5 - 1.3, math.pi * 0.5 + 1.3)
            sample_pose = torch.cat((pitch, yaw), dim=1)
            # sample_pose = poses.deepcopy()
            generator_ori.v_mean = sample_pose[0, 0]
            generator_ori.h_mean = sample_pose[0, 1]
            generator_ori.h_stddev = generator_ori.v_stddev = 0
            
            g_loss, losses_dict, gen_imgs_list = training_step_G(sample_z, sample_pose, real_imgs, 
                zs, real_poses, generator_ddp, ema, ema2, generator_ori_ddp, vgg16, id_loss, 
                optimizer_G, scaler, config, opt, device)

            generator_losses.append(g_loss)
        if rank == 0:
            # interior_step_bar.update(1)
            if (epoch+1) % opt.print_freq == 0:
                elapsed = time.time() - start_time
                rate = elapsed / (epoch + 1.0)
                remaining = (opt.n_epochs - epoch) * rate if rate else 0
                out_str = f"[Experiment: {opt.output_dir}]\n[Epoch: {epoch}/{opt.n_epochs}] [Time: " \
                          f"{total_progress_bar.format_interval(elapsed)} < "\
                          f"{total_progress_bar.format_interval(remaining)}] "
                with open(log_file, 'a') as f:
                    f.write(out_str)
                    f.write("\n")
                print(out_str)

                for loss_key, loss_value in losses_dict.items():
                    with open(log_file, 'a') as f:
                        f.write(f"\t{loss_key}: {loss_value:.4f}\n")
                        print(f"\t{loss_key}: {loss_value:.4f}")

            if (epoch+1) % opt.log_freq == 0:
                for loss_key, loss_value in losses_dict.items():
                    writer.add_scalar(f'G/{loss_key}', loss_value, global_step=epoch)


            # save fixed angle generated images
            if (epoch+1) % opt.sample_interval == 0:
                save_image(gen_imgs_list[2], os.path.join(opt.output_dir, "%06d_debug.png" % epoch),
                           nrow=1, normalize=True, value_range=(-1, 1))
                save_image(gen_imgs_list[0], os.path.join(opt.output_dir, "%06d_debug_reg0.png" % epoch),
                           nrow=1, normalize=True, value_range=(-1, 1))
                save_image(gen_imgs_list[1], os.path.join(opt.output_dir, "%06d_debug_reg1.png" % epoch),
                           nrow=1, normalize=True, value_range=(-1, 1))

            ## save model
            if (epoch+1) % opt.model_save_interval == 0:
                torch.save(ema.state_dict(), os.path.join(opt.output_dir, 'step%06d_ema.pth' % epoch))
                # torch.save(ema2.state_dict(), os.path.join(opt.output_dir, 'step%06d_ema2.pth' % dif_net.step))
                torch.save(generator_ddp.module.state_dict(),
                           os.path.join(opt.output_dir, 'step%06d_generator.pth' % epoch))
    # save_model
    if rank == 0:
        torch.save(ema.state_dict(), os.path.join(opt.output_dir, 'ema.pth'))
        torch.save(ema2.state_dict(), os.path.join(opt.output_dir, 'ema2.pth'))
        torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'generator.pth'))
        torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
        torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
        torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
