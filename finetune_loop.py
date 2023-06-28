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
import fid_evaluation
import datasets
from tqdm import tqdm
from datetime import datetime
import copy
from torch_ema import ExponentialMovingAverage
import pytorch3d
# from loss import *
from torch.utils.tensorboard import SummaryWriter
import pickle, PIL
from PIL import Image
# import utils
import dnnlib

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

def set_facerecon(metadata, device, opt):
    from FaceRecon_Pytorch.models import create_model
    model = create_model(opt, metadata)
    model = model.to(device)
    model.eval()
    return model


def load_models_for_loss(device, opt):
    #for LPIPS loss
    if opt.config.find('FFHQ') >= 0 or opt.config.find('CelebAHQ') >= 0:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)
    elif opt.config.find('CARLA') >= 0 or opt.config.find('CATS') >= 0:   # CATS, CARLA
        import lpips
        vgg16 = lpips.LPIPS(net='vgg').eval().to(device) # closer to "traditional" perceptual loss, when used for optimization
    print("load vgg for LPIPS loss")


    # input image is a [112, 112] size of [-1, 1] range, RGB
    from FaceRecon_Pytorch.models.arcface_torch.backbones import get_model
    face_recog = get_model('r50', fp16=False)
    # face_recog.load_state_dict(torch.load('./FaceRecon_Pytorch/models/arcface_torch/backbone.pth'))
    face_recog.load_state_dict(torch.load(os.path.join(opt.load_dir, 'arcface_torch/backbone.pth'), map_location='cpu'))
    print("load face_recog model for ID loss")
    id_loss = IDLoss(face_recog.eval()).to(device)

    # from flip_loss import HDRFLIPLoss, compute_start_stop_exposures, color_space_transform
    # flip_loss = HDRFLIPLoss().eval().to(device)

    return vgg16, id_loss #, flip_loss


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

    print(f"Loaded pretrained network: {opt.load_dir}")
    if opt.load_dir != '':
        generator.load_state_dict(torch.load(os.path.join(opt.load_dir, opt.pretrain_model, 'generator.pth'), map_location='cpu'))

    generator = generator.to(device)

    if opt.load_dir != '':
        print(f"loaded ema network!")
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
        ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)
        # ema.load_state_dict(torch.load(os.path.join(opt.load_dir, opt.pretrain_model, 'ema.pth'), map_location=device))
        # ema.copy_to(generator.parameters())

        ema = torch.load(os.path.join(opt.load_dir, opt.pretrain_model, 'ema.pth'), map_location=device)
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

    # if opt.load_dir != '':
    #     state_dict = torch.load(os.path.join(opt.load_dir, 'step%06d_optimizer_G.pth'%opt.set_step), map_location='cpu')
    #     optimizer_G.load_state_dict(state_dict)
    
    return optimizer_G

def set_optimizer_G_only(generator_ddp, config, opt):
    if 'equal_lr' not in metadata:
        metadata['equal_lr'] = 5e-2
    if 'sample_lr' not in metadata:
        metadata['sample_lr'] = 1

    optimizer_all = torch.optim.Adam(list(generator_ddp.parameters()),
                                     lr=metadata['gen_lr'], betas=metadata['betas'],
                                     weight_decay=metadata['weight_decay'])
    # mapping_network
    # print(list(generator_ddp.parameters()))
    # import ipdb; ipdb.set_trace()
    # optimizer_all = torch.optim.Adam(list(generator_ddp.module.network.parameters()) +
    #                                  list(generator_ddp.module.output_sigma.parameters()) +
    #                                  list(generator_ddp.module.color_layer_sine.parameters()) +
    #                                  list(generator_ddp.module.output_color.parameters()) ,
    #                                  lr=metadata['gen_lr'], betas=metadata['betas'],
    #                                  weight_decay=metadata['weight_decay'])
    # if opt.load_dir != '':
    #     print("loaded optimizers")
    #     optimizer_all.load_state_dict(
    #         torch.load(os.path.join(opt.load_dir, 'optimizer_all.pth'), map_location='cpu'))

    return optimizer_all


def generate_3DMM_train_data(rays_points, points_exp, deform_ref, opt, faces_exp=None):
    rays_points = rays_points
    device = rays_points.device
    bs, N_rays, N_steps, _ = rays_points.size()
    rays_points = rays_points.reshape(bs, -1, 3)
    _, N, _ = points_exp.size()
    N_random = int(N * opt.sample_3dmm)
    random_indices = torch.randperm(N, device=device)[:N_random]
    points_exp = points_exp[:, random_indices, :]
    distance, idxs, _ = pytorch3d.ops.knn_points(rays_points.float(), points_exp.float(), K=1)

    deform_ref_gather = torch.gather(deform_ref, -2, idxs.expand(-1, -1, 3))
    # print("distance", distance.size(), idxs.size())
    valid = distance.expand(-1, -1, 3) < opt.gen_points_threshold

    valid_all = valid.view(bs, N_rays, N_steps, 3)

    deform_ref_gather[~valid] = 0
    deform_gt = deform_ref_gather.view(bs, N_rays, N_steps, 3)

    # ------------- set invalid distance as -1
    distance = distance / opt.gen_points_threshold  # normalize to 0-1
    distance[distance > 1.0] = 1.0
    # 1.0 represents too far, invalid, 0.0 represent close, valid
    confidence = 1.0 - distance
    confidence = confidence.view(bs, N_rays, N_steps, 1)
    # print("confidence max", confidence.max(), confidence.min())
    on_surface_input_points = rays_points.clone()
    on_surface_input_points[~valid] = 0
    return deform_gt, on_surface_input_points, confidence, valid_all.to(int)


# def training_step_G(sample_freq, sample_phase, sample_id, sample_exp, sample_noise,
#     z_id, z_exp, noise, input_imgs, generator_ddp, ema, ema2, deform_ddp,
#     face_recon_model, vae_net_id, vae_net_exp, imbalance_mse, depth_loss, optimizer_G,
#     alpha, scaler, metadata, opt, device, vgg16, id_loss, generator_ori_ddp=None,
#     metadata2=None):

def training_step_G(sample_z, sample_pose, input_imgs, zs, real_poses, generator_ddp, ema, ema2, 
    generator_ori_ddp,facerecon_model, vgg16, id_loss, optimizer_G, scaler, config, opt, device):
    batch_split = 1
    if opt.load_mat:
        real_imgs = input_imgs[:, :3, :, :]
        mat_imgs = input_imgs[:, 3:, :, :]
    else:
        real_imgs = input_imgs
    bs = zs.size()[0]
    split_batch_size = zs.shape[0] // batch_split  # minibatch split for memory reduction
    img_size = input_imgs.size(-1)

    if opt.loc_reg:
        # --------------------------- interpolate zs and sampled z ---------------------------------
        # regulizer_alpha = torch.rand(1, device=device) * opt.regulizer_alpha + 1.0
        interpolation_direction = sample_z - zs
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        # result_zs = zs + regulizer_alpha * interpolation_direction / interpolation_direction_norm
        result_zs = zs + opt.regulizer_alpha * interpolation_direction / interpolation_direction_norm

    gen_imgs_list = []
    losses_dict = {}
    for split in range(batch_split):
        g_loss = 0
        with torch.cuda.amp.autocast():
            subset_z = zs[split * split_batch_size:(split+1) * split_batch_size]
            generator_ddp.module.get_avg_w()
            gen_imgs, gen_bef_aggr, _ = generator_ddp(subset_z, **config['camera'], detailed_output=True, truncation_psi=opt.psi)
            # gen_imgs = generator_ddp(subset_z, **config['camera'], truncation_psi=opt.psi)[0]

            # --------------------------- loss constraint-----------------------            
            if opt.lambda_id > 0:
                id_l = id_loss(gen_imgs, real_imgs).mean() * opt.lambda_id
                g_loss += id_l
                losses_dict['id_l'] = id_l
            if opt.load_mat and opt.lambda_bg_sigma > 0:
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
                if opt.config.find('FFHQ') >= 0 or opt.config.find('CelebAHQ') >= 0:
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

                elif opt.config.find('CARLA') >= 0 or opt.config.find('CATS') >= 0:   # CATS, CARLA
                    perceptual_loss = vgg16(gen_imgs, real_imgs).mean() * opt.lambda_perceptual
                    # perceptual_loss += vgg16(gen_imgs_d2, real_imgs_d2).mean() * opt.lambda_perceptual
                    # perceptual_loss += vgg16(gen_imgs_d4, real_imgs_d4).mean() * opt.lambda_perceptual
                    # perceptual_loss = perceptual_loss / 3.0

                g_loss += perceptual_loss
                losses_dict['perceptual_loss'] = perceptual_loss
            if opt.lambda_sparse > 0:
                    l1_sparse = torch.sum(rgb_sigma[..., 3].norm(1, dim=(2))) / rgb_sigma.size(0) / rgb_sigma.size(1) * opt.lambda_sparse
                    g_loss += l1_sparse
                    losses_dict['sparse'] = l1_sparse

            # --------------------------- loc_regularization -----------------------
            if opt.loc_reg:
                ## ori G
                subset_sample_z = result_zs[split * split_batch_size:(split + 1) * split_batch_size]
                with torch.no_grad():
                    generator_ori_ddp.module.get_avg_w()
                    sampled_img, sampled_details, gen_positions = generator_ori_ddp(subset_sample_z, **config['camera'], img_size=128, detailed_output=True, truncation_psi=opt.psi)
                ## finetuned G
                output_updated, details_updated, _ = generator_ddp(subset_sample_z, **config['camera'], img_size=128, camera_pos=gen_positions, detailed_output=True, truncation_psi=opt.psi)

                # torch.cuda.empty_cache()
                # detail = {
                #     'points': all_points.reshape(batchsize, img_size, img_size, step, 3),
                #     'outputs': all_outputs.reshape(batchsize, img_size, img_size, step, 4),
                #     'z_vals': all_z_vals.reshape(batchsize, img_size, img_size, step, 1),
                #     'depth': depth.reshape(batchsize, img_size, img_size, 1, 1),
                #     'weights': weights.reshape(batchsize, img_size, img_size, step, 1),
                # }

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
                    if opt.config.find('FFHQ') >= 0 or opt.config.find('CelebAHQ') >= 0:
                        gen_features = vgg16(127.5 * (output_updated + 1), resize_images=False, return_lpips=True)
                        real_features = vgg16(127.5 * (sampled_img + 1), resize_images=False, return_lpips=True)
                        reg_perceptual_loss = ((1000 * gen_features - 1000 * real_features) ** 2).mean() * opt.lambda_perceptual
                    elif opt.config.find('CARLA') >= 0 or opt.config.find('CATS') >= 0:   # CATS, CARLA
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


def staged_forward(z_exp, z_id, noise, generator_ddp, deform_ddp, bfm_model, vae_net_id,
    vae_net_exp, stage, alpha, metadata, opt, truncated_frequencies, truncated_phase_shifts):
    device = z_exp.device
    img_size = metadata['img_size']
    batch_size = z_exp.shape[0]
    neutral_face_flag = False

    split_batch_size = z_exp.shape[0] // metadata['batch_split']  # minibatch split for memory reduction
    # batch split - the number of splited batches
    with torch.no_grad():
        pixels_all = []
        depth_all = []
        pose_all = []
        intersections_deform_all = []
        intersections_canonic_all = []
        is_valid_all = []

        shape_wexp_ori, lm_wexp_ori, shape_wexp_t, lm_wexp_t = bfm_model.compute_bfm_mesh(z_id, z_exp)
        shape_woexp_ori, lm_woexp_ori, shape_woexp_t, lm_woexp_t = bfm_model.compute_bfm_mesh(z_id, None)
        deform_all = shape_woexp_t - shape_wexp_t

        for split in range(metadata['batch_split']):
            subset_z_exp = z_exp[split * split_batch_size:(split + 1) * split_batch_size]
            subset_z_id = z_id[split * split_batch_size:(split + 1) * split_batch_size]
            subset_noise = noise[split * split_batch_size:(split + 1) * split_batch_size]
            subset_face_withexp = shape_wexp_t[split * split_batch_size:(split + 1) * split_batch_size]
            subset_face_woexp = shape_woexp_t[split * split_batch_size:(split + 1) * split_batch_size]
            subset_deform_ref_3dmm = deform_all[split * split_batch_size:(split + 1) * split_batch_size]

            # ------------------------------------------ obtain 3dmm neutral face here-------------------------------------------
            t = time.time()

            if not opt.psi == 1:
                generator_ddp.module.generate_avg_frequencies(vae_net_id, vae_net_exp)
                avg_frequencies = generator_ddp.module.avg_frequencies
                avg_phase_shifts = generator_ddp.module.avg_phase_shifts
                truncated_frequencies = avg_frequencies + opt.psi * (truncated_frequencies - avg_frequencies)
                truncated_phase_shifts = avg_phase_shifts + opt.psi * (truncated_phase_shifts - avg_phase_shifts)
            # else:
            #     truncated_frequencies = raw_frequencies
            #     truncated_phase_shifts = raw_phase_shifts

            with torch.no_grad():
                wp_sample_deform, wp_inter_back_deform, levels, w_ray_origins, w_ray_directions, pitch, yaw, _ = generator_ddp.module.generate_points(
                    subset_z_exp.size()[0], subset_z_exp.device, **metadata)
            t = time.time()
            deform_ref, on_surface_input_points, confidence, near_index = generate_3DMM_train_data(wp_sample_deform,
                                                                                                   subset_face_withexp,
                                                                                                   subset_deform_ref_3dmm,
                                                                                                   opt,
                                                                                                   faces_exp=bfm_model.facemodel.face_buf)
            t2 = time.time()
            w_points_sample_deform_grad, w_vec_deform2canonic, _, correction_density, correction_rgb, gen_positions, output, intersections_deform, intersections_canonical, \
            intersections_density_correct, intersections_rgb_correct, is_valid = \
                generator_ddp.forward(subset_z_id, subset_z_exp, subset_noise, \
                                      wp_sample_deform, wp_inter_back_deform, levels, w_ray_origins, w_ray_directions,
                                      pitch, yaw, \
                                      deform_ref, confidence, neutral_face_flag, deform_ddp, alpha, metadata, \
                                      freq=truncated_frequencies, phase=truncated_phase_shifts, stage_forward_flag=True)

            gen_imgs, depth, weights, transparency, _,_,_ = output
            pixels_all.append(gen_imgs)

            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous()
            depth_all.append(depth_map)
            gen_positions = torch.cat([pitch, yaw], -1)
            pose_all.append(gen_positions)

            intersections_deform_all.append(intersections_deform)
            intersections_canonic_all.append(intersections_canonical)
            is_valid_all.append(is_valid)
        pixels_all_cat = torch.cat([p for p in pixels_all], dim=0)  # 16 x 64 x 64 x 3
        pixels_all_cat = pixels_all_cat.cpu()
        depth_all_cat = torch.cat([p for p in depth_all], dim=0)
        depth_all_cat = depth_all_cat.cpu()

        pose_all_cat = torch.cat([p for p in pose_all], dim=0)

        intersections_deform_cat = torch.cat([p for p in intersections_deform_all], dim=0)
        intersections_canonic_cat = torch.cat([p for p in intersections_canonic_all], dim=0)
        is_valid_all_cat = torch.cat([p for p in is_valid_all], dim=0)

        return pixels_all_cat, depth_all_cat, pose_all_cat, intersections_deform_cat, intersections_canonic_cat, is_valid_all_cat, neutral_face_flag


def training_process(rank, world_size, opt, device):
    # torch.autograd.set_detect_anomaly(True)
    # --------------------------------------------------------------------------------------
    # extract training config
    config = getattr(configs, opt.config)
    # if opt.patch_split is not None:
    #     config['process']['kwargs']['patch_split'] = opt.patch_split
    
    if rank == 0:
        # print(metadata)
        log_dir = opt.output_dir + '/tensorboard/'
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir, 0)

    # --------------------------------------------------------------------------------------
    # set amp gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    # if opt.load_dir != '':
    #     if not config['global'].get('disable_scaler', False):
    #         scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'step%06d_scaler.pth'%opt.set_step)))

    if config['global'].get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)


    # --------------------------------------------------------------------------------------
    # set LPIPS loss and id loss
    vgg16, id_loss = load_models_for_loss(device, opt)

    # --------------------------------------------------------------------------------------
    # set the GRAM generator 
    generator, ema, ema2 = set_generator(config, device, opt)
    facerecon_model = set_facerecon(config, device, opt)
    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    # generator_ddp = DDP(generator, device_ids=[rank])
    generator = generator_ddp.module
    if opt.config.find('FFHQ') >= 0 or opt.config.find('CelebAHQ') >= 0:
        generator.renderer.lock_view_dependence = True
    elif opt.config.find('CATS') >= 0:   # CATS
        generator.renderer.lock_view_dependence = True
    elif opt.config.find('CARLA') >= 0:   # CARLA
        generator.renderer.lock_view_dependence = False

    if rank == 0:
        # for name, param in generator_ddp.named_parameters():
        #     print(f'{name:<{96}}{param.shape}')
        total_num = sum(p.numel() for p in generator_ddp.parameters())
        trainable_num = sum(p.numel() for p in generator_ddp.parameters() if p.requires_grad)
        print('G: Total ', total_num, ' Trainable ', trainable_num)        

    if opt.loc_reg:
        generator_ori, _, _ = set_generator(config, device, opt)
        generator_ori_ddp = DDP(generator_ori, device_ids=[rank], find_unused_parameters=True)
        generator_ori = generator_ori_ddp.module
        # generator_ori.set_device(device)
        generator_ori.eval()
        # generator_ori_ddp.module.generate_avg_frequencies(vae_net_id, vae_net_exp)
        # avg_freq = generator_ori_ddp.module.avg_frequencies
        # avg_phase = generator_ori_ddp.module.avg_phase_shifts
        if opt.config.find('FFHQ') >= 0 or opt.config.find('CelebAHQ') >= 0:
            generator_ori.renderer.lock_view_dependence = True
        elif opt.config.find('CATS') >= 0:   # CATS
            generator_ori.renderer.lock_view_dependence = True
        elif opt.config.find('CARLA') >= 0:   # CARLA
            generator_ori.renderer.lock_view_dependence = False


    # --------------------------------------------------------------------------------------
    # set optimizers
    # generator.set_device(device)
    # facerecon_model.set_device(device)

    optimizer_G = set_optimizer_G(generator_ddp, config, opt)
    # optimizer_G = set_optimizer_G_only(generator_ddp, metadata, opt)

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

    # disable bar display: disable = True
    total_progress_bar = tqdm(total=opt.n_epochs, desc="Total progress", dynamic_ncols=True, disable=True)
    # total_progress_bar.update(dif_net.epoch)  # Keeps track of progress to next stage
    # interior_step_bar = tqdm(dynamic_ncols=True)

    torch.manual_seed(rank)
    # torch.manual_seed(3)

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
            if opt.loc_reg:
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
                
                # # truncated freq & phase shift
                # with torch.no_grad():
                #     sample_z = torch.cat([sample_id, sample_noise], dim=1)
                    # sample_freq, sample_phase = generator_ori_ddp.module.siren.mapping_network(sample_z)
                    # if not opt.psi == 1:
                    #     generator_ori_ddp.module.generate_avg_frequencies(vae_net_id, vae_net_exp)
                    #     avg_freq = generator_ori_ddp.module.avg_frequencies
                    #     avg_phase = generator_ori_ddp.module.avg_phase_shifts
                    #     sample_freq = avg_freq + opt.psi * (sample_freq - avg_freq)
                    #     sample_phase = avg_phase + opt.psi * (sample_phase - avg_phase)

                g_loss, losses_dict, gen_imgs_list = training_step_G(sample_z, sample_pose, real_imgs, 
                    zs, real_poses, generator_ddp, ema, ema2, generator_ori_ddp,
                    facerecon_model, vgg16, id_loss, optimizer_G, scaler, config, opt, device)
            else:
                g_loss, losses_dict, gen_imgs_list = training_step_G(None, None, real_imgs, 
                    zs, real_poses, generator_ddp, ema, ema2, None,
                    facerecon_model, vgg16, id_loss, optimizer_G, scaler, config, opt, device)

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

            # if epoch % opt.save_mesh == 0:
            #     ## save image corresponding to 3d mesh

            #     ## visualize weighted points and bfm_noexp and landmark points

            #     weighted_points_vis = weighted_points[0, ...].detach().cpu().numpy().reshape(-1, 3)
            #     our_mesh = trimesh.Trimesh(weighted_points_vis)
            #     our_mesh.export(os.path.join(opt.output_dir, "%06d_weighted_points_ours.ply" % epoch))

            #     triangles = facerecon_model.facemodel.face_buf.detach().cpu().numpy()
            #     bfm_points = face_withexp[0, ...].detach().cpu().numpy().reshape(-1, 3)
            #     bfm_mesh = trimesh.Trimesh(bfm_points, np.array(triangles))
            #     bfm_mesh.export(os.path.join(opt.output_dir, "%06d_facewithexp.ply" % epoch))

            #     # save landmark3D
            #     bfm_landmark_vis = bfm_landmark_list[0][0, ...].detach().cpu().numpy().reshape(-1, 3)
            #     bfm_landmark_mesh = trimesh.Trimesh(bfm_landmark_vis)
            #     bfm_landmark_mesh.export(
            #         os.path.join(opt.output_dir, "%06d_landmark3D_bfm.ply" % epoch))

            #     # vis predicted bfm mesh and landmark
            #     shape_wexp_ori_pred, lm_wexp_ori_pred, shape_wexp_t_pred, lm_wexp_t_pred = pred_bfm_list[0]
            #     shape_pred = shape_wexp_t_pred[0, ...].detach().cpu().numpy().reshape(-1, 3)
            #     pred_bfm_mesh = trimesh.Trimesh(shape_pred, np.array(triangles))
            #     pred_bfm_mesh.export(os.path.join(opt.output_dir, "%06d_face_pred.ply" % epoch))

                # pred_landmark_vis = lm_wexp_t_pred[0, ...].detach().cpu().numpy().reshape(-1, 3)
                # pred_landmark_mesh = trimesh.Trimesh(pred_landmark_vis)
                # pred_landmark_mesh.export(
                #     os.path.join(opt.output_dir, "%06d_landmark3D_pred.ply" % discriminator.step))

            # save fixed angle generated images
            if (epoch+1) % opt.sample_interval == 0:
                if opt.loc_reg:
                    save_image(gen_imgs_list[2], os.path.join(opt.output_dir, "%06d_debug.png" % epoch),
                               nrow=1, normalize=True, value_range=(-1, 1))
                    save_image(gen_imgs_list[0], os.path.join(opt.output_dir, "%06d_debug_reg0.png" % epoch),
                               nrow=1, normalize=True, value_range=(-1, 1))
                    save_image(gen_imgs_list[1], os.path.join(opt.output_dir, "%06d_debug_reg1.png" % epoch),
                               nrow=1, normalize=True, value_range=(-1, 1))
                else:
                    save_image(gen_imgs_list[0], os.path.join(opt.output_dir, "%06d_debug.png" % epoch),
                               nrow=1, normalize=True, value_range=(-1, 1))

                ## debug
                # if poses[0, 1] < 1.0:
                #     tilted_pose = [0.5, 1.0]
                # elif poses[0, 1] > 2.0:
                #     tilted_pose = [-0.5, -1.0]
                # else:
                #     tilted_pose = [-0.5, 0.5]

                # with torch.no_grad():
                #     with torch.cuda.amp.autocast():

                #         gen_imgs1 = generator_ddp(fixed_z, **config['camera'], truncation_psi=opt.psi)[0]

                #         generator.h_mean = poses[0, 1] + tilted_pose[0]
                #         gen_imgs2 = generator_ddp(fixed_z, **config['camera'], truncation_psi=opt.psi)[0]


                #         generator.h_mean = poses[0, 1] + tilted_pose[1]
                #         gen_imgs3 = generator_ddp(fixed_z, **config['camera'], truncation_psi=opt.psi)[0]

                # # save image
                # output_name = "%06d_fixed.png" % epoch
                # save_image(gen_imgs1[:25], os.path.join(opt.output_dir, output_name), nrow=4, normalize=True,
                #            value_range=(-1, 1))

                # output_name = "%06d_tilted_1.png" % epoch
                # save_image(gen_imgs2[:25], os.path.join(opt.output_dir, output_name), nrow=4, normalize=True,
                #            value_range=(-1, 1))

                # output_name = "%06d_tilted_2.png" % epoch
                # save_image(gen_imgs3[:25], os.path.join(opt.output_dir, output_name), nrow=4, normalize=True,
                #            value_range=(-1, 1))

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

    # --------------------------------------------------------------------------------------
    # FID evaluation
    # if opt.eval_freq > 0 and (discriminator.step+1) % opt.eval_freq == 0:
    #     print("start evaluation")
    #     generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')

    #     copied_metadata_eval = copy.deepcopy(metadata)
    #     print("copied metadata", copied_metadata_eval)
    #     if rank == 0:
    #         real_dir, dataloader = fid_evaluation.setup_evaluation(copied_metadata_eval['dataset'], dataset, generated_dir, opt, target_size=128, metadata=copied_metadata_eval)
    #         dist.barrier()
    #         fid_evaluation.output_images(dataloader, generator_ddp, dif_net_ddp, copied_metadata_eval, rank, world_size, generated_dir, alpha)
    #         dist.barrier()
    #     if rank == 0:
    #         fid = fid_evaluation.calculate_fid(copied_metadata_eval['dataset'], generated_dir, target_size=128)
    #         with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
    #             f.write(f'\n{discriminator.step}:{fid}')
