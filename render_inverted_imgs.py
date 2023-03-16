# from asyncio import FIRST_COMPLETED
# from logging import shutdown
import os

import ipdb
from matplotlib.pyplot import prism
import numpy as np
import math
from collections import deque

from yaml import parse
import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import torchvision.transforms as transforms
import importlib
import time 
import glob, shutil
from scipy.io import loadmat
import copy
from generators import generators
import configs

# from generators import generators_neutex as generators
# from discriminators import discriminators
# from siren import siren
# import fid_evaluation
# import datasets
# import curriculums
from tqdm import tqdm
# from datetime import datetime
# import copy, plyfile
from torch_ema import ExponentialMovingAverage
# import pytorch3d
# from loss import *
from torch.utils.tensorboard import SummaryWriter
# from torch_ema import ExponentialMovingAverage
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# import FaceRecon_Pytorch.util.util as util
# device = torch.device('cuda') #if torch.cuda.is_available() else 'cpu')
# from loss import * 
from PIL import Image 
import skvideo
# skvideo.setFFmpegPath("/usr/bin/")
import skvideo.io
from skvideo.io import FFmpegWriter
import PIL.ImageDraw as ImageDraw

import plyfile
import mrcfile
import skimage.measure


def load_models(opt, config, device):
    generator_args = {}
    if 'representation' in config['generator']:
        generator_args['representation_kwargs'] = config['generator']['representation']['kwargs']
    if 'renderer' in config['generator']:
        generator_args['renderer_kwargs'] = config['generator']['renderer']['kwargs']
    generator = getattr(generators, config['generator']['class'])(
        **generator_args,
        **config['generator']['kwargs']
    )
    print(opt.generator_file)
    generator.load_state_dict(torch.load(os.path.join(opt.generator_file), map_location='cpu'),strict=False)
    generator = generator.to('cuda')
    generator.eval()


    # ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    # ema_file = torch.load(os.path.join(opt.generator_file.replace('generator', 'ema2')), map_location='cpu')
    # ema_file['collected_params'] = ema_file['shadow_params']
    # ema.load_state_dict(ema_file)
    # parameters = [p for p in generator.parameters() if p.requires_grad]
    # ema.copy_to(parameters)

    try:
        ema = torch.load(os.path.join(opt.generator_file.replace('generator', 'ema')), map_location='cpu')
        parameters = [p for p in generator.parameters() if p.requires_grad]
        ema.copy_to(parameters)
    except:
        pass


    return generator

def read_pose_ori(name,flip=False):
    P = loadmat(name)['angle']
    P_x = -(P[0,0] - 0.1) + math.pi/2
    if not flip:
        P_y = P[0,1] + math.pi/2
    else:
        P_y = -P[0,1] + math.pi/2


    P = torch.tensor([P_x,P_y],dtype=torch.float32)

    return P

def read_pose_npy(name,flip=False):
    P = np.load(name)
    P_x = P[0] + 0.14
    if not flip:
        P_y = P[1]
    else:
        P_y = -P[1] + math.pi

    P = torch.tensor([P_x,P_y],dtype=torch.float32)

    return P


def transform_matrix_to_camera_pos(c2w,flip=False):
    """
    Get camera position with transform matrix

    :param c2w: camera to world transform matrix
    :return: camera position on spherical coord
    """

    c2w[[0,1,2]] = c2w[[1,2,0]]
    pos = c2w[:, -1].squeeze()
    radius = float(np.linalg.norm(pos))
    theta = float(np.arctan2(-pos[0], pos[2]))
    phi = float(np.arctan(-pos[1] / np.linalg.norm(pos[::2])))
    theta = theta + np.pi * 0.5
    phi = phi + np.pi * 0.5
    if flip:
        theta = -theta + math.pi
    P = torch.tensor([phi,theta],dtype=torch.float32)
    return P

def read_latents_txt_fq(name, device="cpu"):
    # load the latent codes for id, expression and so on.
    '''
        the data structure of freq_phase inversion
        latents: (5376,)
        freq: input freq for PiGAN, 2304
        phase: input phase for PiGAN, 2304
    '''
    latents = np.loadtxt(name)
    latents = torch.from_numpy(latents).float().unsqueeze(0).to(device)
    freq = latents[:, :2304]
    phase = latents[:, 2304:2304+2304]

    return freq, phase

def read_latents_txt_z(name, device="cpu"):
    '''
        the data structure of z inversion
    '''
    latents = np.loadtxt(name)
    latents = torch.from_numpy(latents).float().unsqueeze(0).to(device)

    return latents

def get_trajectory(type, num_frames, latent_code1, latent_code2=None):
    latent_codes = []
    if type == 'still':
        for pp in range(num_frames):
            latent_codes.append((latent_code1))
    elif type == 'gradual':
        ratio = np.linspace(0, 1.0, num_frames)
        for pp in range(num_frames):
            latent_codes.append((ratio[pp] * latent_code1))
    elif type == 'interpolate':
        # interpolate between two inverted images
        ratio = np.linspace(0, 1.0, num_frames)
        for pp in range(num_frames):
            latent_code_interpolate = ratio[pp] * latent_code2 + (1 - ratio[pp]) * latent_code1
            latent_codes.append(latent_code_interpolate)
    return latent_codes

def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )

    faces = faces[:,::-1]

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
                   
    return samples.unsqueeze(0), voxel_origin, voxel_size


def sample_generator(generator, z, max_batch=100000, voxel_resolution=256, voxel_origin=[0,0,0], cube_length=2.0, psi=0.7):
    head = 0
    samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
    samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
    
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
    transformed_ray_directions_expanded[..., -1] = -1
    
    generator.get_avg_w()
    with torch.no_grad():
        while head < samples.shape[1]:
            coarse_output = generator._volume(z, truncation_psi=psi)(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head])

            sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
            head += max_batch
    
    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    
    return sigmas, voxel_origin, voxel_size

def sample_generator_with_freq_phase(generator, freq, phase, max_batch=100000, voxel_resolution=256, voxel_origin=[0,0,0], cube_length=2.0):
    head = 0
    samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
    samples = samples.to(freq.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=freq.device)
    
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=freq.device)
    transformed_ray_directions_expanded[..., -1] = -1
    
    # generator.get_avg_w()
    with torch.no_grad():
        while head < samples.shape[1]:
            coarse_output = generator._volume_with_frequencies_phase_shifts(freq, phase)(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head])

            sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
            head += max_batch
    
    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    
    return sigmas, voxel_origin, voxel_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_file', type=str, default='../pretrained_models/FFHQ_default/generator.pth')
    parser.add_argument('--output_dir', type=str, default='../exp_gram/inversion_ffhq/')
    parser.add_argument('--img_dir', type=str, default='../../Dataset/NeRFGAN/image256_align_new_mirror_wo_t')
    parser.add_argument('--mat_dir', type=str, default='../../Dataset/NeRFGAN/ffhq_pose_align_new_mirror')
    parser.add_argument('--config', type=str, default='FFHQ_default')
    parser.add_argument('--target_emb_dir', type=str, default='../exp_gram/inversion/')
    parser.add_argument('--target_name', type=str, default=None)
    parser.add_argument('--max_batch_size', type=int, default=1200000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--name', type=str, default='render', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--psi', type=float, default=1)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trajectory', type=str, default='front', help='still, front, orbit')
    parser.add_argument('--z_trajectory', type=str, default='still', help='still, gradual, interpolate')
    parser.add_argument('--freq_trajectory', type=str, default='still', help='still, gradual, interpolate')
    parser.add_argument('--phase_trajectory', type=str, default='still', help='still, gradual, interpolate')
    parser.add_argument('--opt_param', type=str, default="freq_phase", help="select what parameters to optimize, e.g., freq_phase, freq2_phase2 or id_exp_noise")
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
    parser.add_argument('--gen_video', action='store_true', help='whether generate video')
    parser.add_argument('--cube_size', type=float, default=0.3)
    parser.add_argument('--voxel_resolution', type=int, default=256)
    parser.add_argument('--use_depth', action='store_true', help='whether use depth loss for geomotry generation')
    parser.add_argument('--white_bg', action='store_true', help='whether use white background')
    opt = parser.parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = getattr(configs, opt.config)
    if opt.white_bg:
        config['generator']['renderer']['kwargs']['white_back'] = True
        config['generator']['renderer']['kwargs']['background'] = False
    ## load models
    generator = load_models(opt, config, device)

    ## load data
    if opt.config.find('FFHQ') >= 0:
        generator.renderer.lock_view_dependence = True
        img_size = config['global']['img_size']
    elif opt.config.find('CelebAHQ') >= 0:
        generator.renderer.lock_view_dependence = True
        img_size = config['global']['img_size']
    elif opt.config.find('CATS') >= 0:   # CATS
        generator.renderer.lock_view_dependence = True
        img_size = config['global']['img_size']
    elif opt.config.find('CARLA') >= 0:   # CARLA
        generator.renderer.lock_view_dependence = False
        img_size = config['global']['img_size']

    if opt.target_name == "all":
        if opt.target_emb_dir.find("_Z") >= 0:
            target_emb_name = "*/00999_.txt"
        else:
            target_emb_name = "*/01999_.txt"
    else:
        if opt.target_emb_dir.find("_Z") >= 0:
            target_emb_name = f"{opt.target_name}/00999_.txt"
        else:
            target_emb_name = f"{opt.target_name}/01999_.txt"
        # target_emb_name = f"{opt.target_name}/01999_.txt" #"CoreView_142_cam2_29985799"
        # target_emb_name = f"{opt.target_name}/01999_avgFreqPhase.txt"
        

    invert_file_targets = sorted(glob.glob(os.path.join(opt.target_emb_dir, target_emb_name)))
    for invert_file_target in invert_file_targets:
        print(f"Rendering for {invert_file_target}")
        extract_shape = False
        ## load pose, inverted latent code
        target_name = invert_file_target.split("/")[-2]

        if opt.config.find('FFHQ') >= 0 or opt.config.find('CelebAHQ') >= 0:
            mat_target = os.path.join(opt.mat_dir, f"{target_name}.mat")
            pose = read_pose_ori(mat_target, flip=False)
        elif opt.config.find('CATS') >= 0:   # CATS
            mat_target = os.path.join(opt.mat_dir, f"{target_name}_pose.npy")
            pose = read_pose_npy(mat_target, flip=False)
        elif opt.config.find('CARLA') >= 0:   # CARLA
            mat_target = os.path.join(opt.mat_dir, f"{target_name}_extrinsics.npy")
            pose = transform_matrix_to_camera_pos(np.load(mat_target))
        else:
            raise

        # import ipdb;ipdb.set_trace()

        if opt.trajectory == 'still_pose':
            num_frames = 1
            extract_shape = True
        else:
            num_frames = 100
        ## set freqs & phases or z
        if opt.opt_param == "freq_phase":
            freq, phase = read_latents_txt_fq(invert_file_target, device=device)
            freqs = get_trajectory(opt.freq_trajectory, num_frames, freq)
            phases = get_trajectory(opt.phase_trajectory, num_frames, phase)
        elif opt.opt_param == "z":
            z = read_latents_txt_z(invert_file_target, device=device)
            zs = get_trajectory(opt.z_trajectory, num_frames, z)

        ## set trajectory
        if opt.trajectory == 'still_front':
            trajectory = []
            pose_ratio = np.linspace(0, 1, num_frames)
            for t in np.linspace(0, 1, num_frames):
                ## frontal face
                fixed_t = pose_ratio[0]  # t=pose_ratio[19]
                pitch = math.pi / 2  # 0.2 * np.cos(t * 2 * math.pi) + math.pi / 2
                yaw = 0.4 * np.sin(fixed_t * 2 * math.pi) + math.pi / 2
                fov = 12
                trajectory.append((pitch, yaw, fov))
        elif opt.trajectory == 'still_pose':
            ## pose of ori image
            trajectory = []
            for t in np.linspace(0, 1, num_frames):
                pitch = pose[0]
                yaw = pose[1]
                fov = 12
                trajectory.append((pitch, yaw, fov))
        elif opt.trajectory == 'front':
            trajectory = []
            for t in np.linspace(0, 1, num_frames):
                pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi / 2
                yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi / 2
                fov = 12
                trajectory.append((pitch, yaw, fov))
        elif opt.trajectory == 'orbit':
            trajectory = []
            for t in np.linspace(0, 1, num_frames):
                pitch = math.pi / 4
                yaw = t * 2 * math.pi
                fov = config['camera']['fov']
                # fov = curriculum['fov']

                trajectory.append((pitch, yaw, fov))

        ## generate images
        with torch.no_grad():
            flag = True
            images = []
            depths = []

            generator.get_avg_w()
            output_name = os.path.join(opt.output_dir, f"{target_name}_{opt.suffix}.mp4")
            if os.path.exists(output_name):
                continue
            writer = FFmpegWriter(output_name, outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'}, verbosity=10)
            frames = []

            cnt_output_dir = os.path.join(opt.output_dir, '%s_%s/'%(target_name, opt.suffix))
            os.makedirs(cnt_output_dir, exist_ok=True)

            for frame_idx in range(num_frames):
                pitch, yaw, fov = trajectory[frame_idx]

                # import ipdb; ipdb.set_trace()
                generator.h_mean = yaw
                generator.v_mean = pitch
                generator.h_stddev = generator.v_stddev = 0

                # generate img
                # import ipdb; ipdb.set_trace()
                if opt.opt_param == "freq_phase":
                    freq = freqs[frame_idx]
                    phase = phases[frame_idx]
                    tensor_img = generator.forward_with_frequencies_phase_shifts(freq, phase, **config['camera'])[0]

                    if extract_shape:
                        voxel_grid, voxel_origin, voxel_size = sample_generator_with_freq_phase(
                            generator, freq, phase, cube_length=opt.cube_size, voxel_resolution=opt.voxel_resolution)

                elif opt.opt_param == "z":
                    z = zs[frame_idx]
                    tensor_img = generator(z, **config['camera'], truncation_psi=opt.psi)[0]

                    if extract_shape:
                        voxel_grid, voxel_origin, voxel_size = sample_generator(
                            generator, z, cube_length=opt.cube_size, voxel_resolution=opt.voxel_resolution)

                # # metadata['lock_view_dependence'] = False
                # tensor_img = staged_forward(freq, phase, z_id, z_exp, noise, generator, dif_model, bfm_model, vae_net_id,
                #                          vae_net_exp, stage=metadata['img_size'], alpha=1, metadata=metadata, opt=opt)

                # img = Image.new('L', (opt.image_size, opt.image_size), 0)
                # bs, _, img_size, _ = tensor_img.size()
                save_image(tensor_img, os.path.join(cnt_output_dir, f"{target_name}_{frame_idx}_.png"), normalize=True,range=(-1,1))
                frames.append(tensor_to_PIL(tensor_img))
                ## save shape
                if extract_shape:
                    l = 5
                    try:        
                        convert_sdf_samples_to_ply(voxel_grid, voxel_origin, voxel_size, 
                            os.path.join(opt.output_dir,f'{target_name}.ply'), level=l)
                        # with mrcfile.new_mmap(os.path.join(opt.output_dir, f'{target_name}.mrc'), overwrite=True, shape=voxel_grid.shape, mrc_mode=2) as mrc:
                        #     mrc.data[:] = voxel_grid
                    except:
                        continue

            for frame in frames:
                writer.writeFrame(np.array(frame))

            writer.close()
