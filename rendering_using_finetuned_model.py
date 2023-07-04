import os
from matplotlib.pyplot import prism
import numpy as np
import math
from collections import deque

from yaml import parse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
import importlib
import time 
import glob, shutil
from scipy.io import loadmat
import copy
from generators import generators
import configs

from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter

import argparse
from PIL import Image 
import skvideo
skvideo.setFFmpegPath("/usr/bin/")
# import skvideo.io
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_file', type=str, default='experiments/gram/finetuned_model/subject_name/generator.pth')
    parser.add_argument('--target_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='experiments/gram/rendering_results/')
    parser.add_argument('--data_img_dir', type=str, default='samples/faces/')
    parser.add_argument('--data_pose_dir', type=str, default='samples/faces/poses/')
    parser.add_argument('--data_emd_dir', type=str, default='experiments/gram/inversion')
    parser.add_argument('--config', type=str, default='FACES_default')
    parser.add_argument('--max_batch_size', type=int, default=1200000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--name', type=str, default='render', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--psi', type=float, default=0.7)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trajectory', type=str, default='front', help='still, front, orbit')
    parser.add_argument('--z_trajectory', type=str, default='still', help='still, gradual, interpolate')
    parser.add_argument('--freq_trajectory', type=str, default='still', help='still, gradual, interpolate')
    parser.add_argument('--phase_trajectory', type=str, default='still', help='still, gradual, interpolate')
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
    generator.renderer.lock_view_dependence = True
    img_size = config['global']['img_size']
    target_emb_name = f"{opt.target_name}/00999_.txt"
    optimized_latents = sorted(glob.glob(os.path.join(opt.data_emd_dir, target_emb_name)))

    for optimized_latent in optimized_latents:
        print(f"Rendering for {optimized_latent}")
        if not os.path.exists(optimized_latent):
            print(f"The file '{optimized_latent}' does not exist.")
            raise

        extract_shape = False
        ## load pose, inverted latent code
        target_name = optimized_latent.split("/")[-2]

        if opt.config.find('FACES') >= 0:
            mat_target = os.path.join(opt.data_pose_dir, f"{target_name}.mat")
            pose = read_pose_ori(mat_target, flip=False)
        elif opt.config.find('CATS') >= 0:   # CATS
            mat_target = os.path.join(opt.data_pose_dir, f"{target_name}_pose.npy")
            pose = read_pose_npy(mat_target, flip=False)
        else:
            raise

        if opt.trajectory == 'still_pose':
            num_frames = 1
            extract_shape = True
        else:
            num_frames = 100
        ## set latent code z
        z = read_latents_txt_z(optimized_latent, device=device)
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
                fov = curriculum['fov']

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

                generator.h_mean = yaw
                generator.v_mean = pitch
                generator.h_stddev = generator.v_stddev = 0

                # generate img
                z = zs[frame_idx]
                tensor_img = generator(z, **config['camera'], truncation_psi=opt.psi)[0]

                if extract_shape:
                    voxel_grid, voxel_origin, voxel_size = sample_generator(
                        generator, z, cube_length=opt.cube_size, voxel_resolution=opt.voxel_resolution)

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
