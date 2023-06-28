import os
import glob, shutil
import torch

from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
from pytorch_fid import fid_score
from tqdm import tqdm
import copy
import argparse

# import pytorch3d 

from generators import generators
import configs
import math

import time
from PIL import Image
import torchvision.transforms as transforms
import dnnlib
import numpy as np
from scipy.io import loadmat
# from arcface.facenet import iresnet18
import torch.nn.functional as F
import torch.nn as nn
import importlib
from torch_ema import ExponentialMovingAverage
import FaceRecon_Pytorch.util.util as util


class IDLoss(nn.Module):
    def __init__(self, facenet):
        super(IDLoss, self).__init__()
        self.facenet = facenet
             
    def forward(self,x,y):
        x = F.interpolate(x,size=[112,112],mode='bilinear')
        y = F.interpolate(y,size=[112,112],mode='bilinear')

        # x = 2*(x-0.5)
        # y = 2*(y-0.5)
        feat_x = self.facenet(x)
        feat_y = self.facenet(y.detach())

        loss = 1 - F.cosine_similarity(feat_x,feat_y,dim=-1)

        return loss

def read_pose(name,flip=False):
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


def load_models(opt, config, device):
    print("loading models...")
    generator_args = {}
    if 'representation' in config['generator']:
        generator_args['representation_kwargs'] = config['generator']['representation']['kwargs']
    if 'renderer' in config['generator']:
        generator_args['renderer_kwargs'] = config['generator']['renderer']['kwargs']
    generator = getattr(generators, config['generator']['class'])(
        **generator_args,
        **config['generator']['kwargs']
    )

    generator.load_state_dict(torch.load(os.path.join(opt.generator_file), map_location='cpu'),strict=False)
    generator = generator.to('cuda')
    generator.eval()
    
    ema = torch.load(os.path.join(opt.generator_file.replace('generator', 'ema')), map_location='cuda')
    parameters = [p for p in generator.parameters() if p.requires_grad]
    ema.copy_to(parameters)

    #for LPIPS loss
    if opt.config == 'FFHQ_default' or opt.config == 'CelebAHQ_default':
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)
    elif opt.config == 'CARLA_default' or opt.config == 'CATS_default':   # CATS, CARLA
        import lpips
        vgg16 = lpips.LPIPS(net='vgg').eval().to(device) # closer to "traditional" perceptual loss, when used for optimization
    else:
        raise

    # input image is a [112, 112] size of [-1, 1] range, RGB
    from FaceRecon_Pytorch.models.arcface_torch.backbones import get_model
    face_recog = get_model('r50', fp16=False)
    face_recog.load_state_dict(torch.load('../pretrained_models/FaceRecon_Pytorch/models/arcface_torch/backbone.pth'))
    face_recog.eval()

    return generator, vgg16, face_recog


# def load_metadata(curriculum):
#     curriculum = getattr(curriculums, opt.curriculum)
#     curriculum['dataset'] = 'FFHQ128'
#     metadata = curriculums.extract_metadata(curriculum, 0)
#     metadata['img_size'] = 128
#     metadata['num_steps'] = 24
#     metadata['batch_size'] = 1
#     metadata['batch_split'] = 1
#     # metadata['h_stddev'] = 0.22
#     # metadata['v_stddev'] = 0.2

#     # metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
#     # metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
#     metadata['fov'] = 12
#     metadata['ray_start'] = 0.88
#     metadata['ray_end'] = 1.12
#     metadata['sample_dist'] = 'gaussian'
#     metadata['h_stddev'] = 0
#     metadata['v_stddev'] = 0
#     metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
#     metadata['psi'] = 1.0

#     metadata['final_num_steps'] = 24
#     metadata['nerf_noise'] = 0
#     metadata['interval_scale'] = 1.
#     metadata['has_back'] = False
#     metadata['phase_noise'] = False
#     metadata['delta_final'] = 1e10
#     metadata['hierarchical_sample'] = 1
#     metadata['train_coarse'] = True
#     metadata['levels_start'] = 23
#     metadata['levels_end'] = 8
#     metadata['use_alpha'] = True
#     metadata['alpha_delta'] = 0.04
#     metadata['num_levels'] = metadata['num_steps'] - 1
#     metadata['hidden_dim_sample'] = 128
#     metadata['lock_view_dependence'] = True
#     metadata['model'] = 'SPATIALSIRENMULTI_NEW'
#     metadata['model_sample'] = 'SPATIALSAMPLERELU'
#     metadata['generator'] = 'ImplicitGenerator3d'
#     return metadata

if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_file', type=str, default='./pretrained_models/FFHQ_default/generator.pth')
    parser.add_argument('--output_dir', type=str, default='../exp_gram/inversion_ffhq/')
    parser.add_argument('--img_dir', type=str, default='../../Dataset/NeRFGAN/image256_align_new_mirror_wo_t/')
    parser.add_argument('--mat_dir', type=str, default='../../Dataset/NeRFGAN/ffhq_pose_align_new_mirror/')
    parser.add_argument('--config', type=str, default='FFHQ_default')
    parser.add_argument('--gpu_type', type=str, default='8000')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument('--num_images', type=int, default=5000, help="the number of inverted images or -1 (all images)")
    parser.add_argument('--name', type=str, default=None, help="specifc image name (e.g. '28606'), or None (will invert all images)")
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
    parser.add_argument('--psi', type=str, default=0.7)
    parser.add_argument('--lambda_perceptual', type=float, default=1)
    parser.add_argument('--lambda_l2', type=float, default=0.01)
    parser.add_argument('--lambda_id', type=float, default=0.01)
    parser.add_argument('--lambda_reg', type=float, default=0.04)

    parser.add_argument('--checkpoints_dir', type=str, default='../pretrained_models/FaceRecon_Pytorch/checkpoints', help='models are saved here')
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
    parser.add_argument('--init_path', type=str, default='../pretrained_models/FaceRecon_Pytorch/checkpoints/init_model/resnet50-0676ba61.pth')
    parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='../pretrained_models/FaceRecon_Pytorch/BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)
    parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
        )
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='./backbone.pth')

    parser.add_argument('--start_iter', type=int, default=2000)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--sv_interval', type=int, default=20)
    parser.add_argument('--vis_loss', type=bool, default=False)
    parser.add_argument('--opt_param', type=str, default="freq_phase", help="select what parameters to optimize, e.g., freq_phase, freq2_phase2 or id_exp_noise")

    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = getattr(configs, opt.config)

    ## load models
    generator, vgg16, face_recog = load_models(opt, config, device)

    from flip_loss import HDRFLIPLoss, compute_start_stop_exposures, color_space_transform
    flip_loss = HDRFLIPLoss().eval().to(device)

    ## load data
    print("---> opt.config: ", opt.config)
    if opt.config.find('FFHQ') >= 0:
        # opt.img_dir='../../Dataset/NeRFGAN/image256_align_new_mirror_wo_t/'
        # opt.mat_dir='../../Dataset/NeRFGAN/ffhq_pose_align_new_mirror/'
        generator.renderer.lock_view_dependence = True
        img_size = config['global']['img_size']
        transform = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    elif opt.config.find('CelebAHQ') >= 0:
        # opt.img_dir='../../Dataset/NeRFGAN/celebahq_test256_align_new_mirror_wo_t/'
        # opt.mat_dir='../../Dataset/NeRFGAN/celebahq_test256_mat/'
        generator.renderer.lock_view_dependence = True
        img_size = config['global']['img_size']
        transform = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    elif opt.config.find('CATS') >= 0:   # CATS
        # opt.img_dir='../../Dataset/NeRFGAN/cats2_256/'
        # opt.mat_dir='../../Dataset/NeRFGAN/cats2/poses/'
        generator.renderer.lock_view_dependence = True
        img_size = config['global']['img_size']
        transform = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    elif opt.config.find('CARLA') >= 0:   # CARLA
        # opt.img_dir='../../Dataset/NeRFGAN/carla128/'
        # opt.mat_dir='../../Dataset/NeRFGAN/carla/carla_poses/'
        generator.renderer.lock_view_dependence = False
        img_size = config['global']['img_size']
        transform = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    else:
        raise
        # print("--->", 'FFHQ')


        # raise Exception("No config")

    ## search all images
    img_fullpaths_all = []
    if opt.name:
        name = opt.name
        img_fullpath = os.path.join(opt.img_dir, f"{opt.name}.png")
        img_fullpaths_all.append(img_fullpath)
    else:
        img_fullpaths_all = sorted(glob.glob(os.path.join(opt.img_dir, f"*.png")))
        img_fullpaths_all = img_fullpaths_all[:opt.num_images]
    img_fullpaths = []
    for imgpath in img_fullpaths_all:
        subject = imgpath.split('/')[-1].split('.')[0]
        inv_path = os.path.join(opt.output_dir, subject, f"{(opt.max_iter-1):05d}_.txt")
        # if not os.path.exists(inv_path):
        if not os.path.exists(inv_path):
            img_fullpaths.append(imgpath)
        else:
            print(f"Ignoring {subject}...")


    ## start optimization
    for img_fullpath in img_fullpaths:
        ## load image and mat file (ori pose)
        print(f"Processing {img_fullpath}...")
        img = Image.open(img_fullpath)
        img = transform(img).cuda()
        img = img.unsqueeze(0)
        name = img_fullpath.split("/")[-1][:-4]
        if opt.config.find('FFHQ') >= 0 or opt.config.find('CelebAHQ') >= 0:
            mat_fullpath = os.path.join(opt.mat_dir, f"{name}.mat")
            pose = read_pose(mat_fullpath)
        elif opt.config.find('CATS') >= 0:   # CATS
            mat_fullpath = os.path.join(opt.mat_dir, f"{name}_pose.npy")
            pose = read_pose_npy(mat_fullpath)
        elif opt.config.find('CARLA') >= 0:   # CARLA
            mat_fullpath = os.path.join(opt.mat_dir, f"{name}_extrinsics.npy")
            pose = transform_matrix_to_camera_pos(np.load(mat_fullpath))
        else:
            raise
        if opt.opt_param == "z_pose":
            ## optimize camera pose
            cuda0 = torch.device('cuda:0')
            h_mean = torch.tensor([pose[1]], requires_grad=True, device=cuda0)
            v_mean = torch.tensor([pose[0]], requires_grad=True, device=cuda0)
            generator.h_stddev = generator.v_stddev = 0
        else:
            ## do not optimize camera pose, directly use pose predicted by the reconstruction network
            generator.h_mean = pose[1]
            generator.v_mean = pose[0]
            generator.h_stddev = generator.v_stddev = 0

        # set output_dir
        output_dir = os.path.join(opt.output_dir, f"{name}")
        os.makedirs(output_dir, exist_ok=True)
        f = open(os.path.join(output_dir, 'logs.txt'), "w")
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(config))
        f.write('\n\n')

        load_prev_file = os.path.join(output_dir, '%05d_%s.txt' % (opt.start_iter-1, opt.suffix))
        if opt.opt_param == "freq_phase":
            patch_split = 2
            generator.get_avg_w()
            avg_freq = generator.representation.rf_network.avg_frequencies # torch.Size([1, 2304])
            avg_phase = generator.representation.rf_network.avg_phase_shifts # torch.Size([1, 2304])
            length = avg_freq.shape[-1]
            # debug
            # invert_file = f"../experiments/inversion_ffhq/00000/01999_avgFreqPhase.txt"
            # invert_file = f"../experiments/inversion/CoreView_142_cam11_29985799/01999_.txt"
            # avg_freq, avg_phase = read_latents_txt(invert_file, device=device)
            if not os.path.exists(load_prev_file):
                start_iter = 0
                # initialize freq and phase with average
                latent_code = torch.cat([avg_freq, avg_phase], dim=-1).detach().clone()
                latent_code.requires_grad = True
                latent_code = latent_code.to(device)
            else:
                start_iter = opt.start_iter
                latents = np.loadtxt(load_prev_file)            
                latent_code = torch.from_numpy(latents).float().unsqueeze(0).to(device)
                latent_code.requires_grad = True
            optimizer = torch.optim.Adam([latent_code], lr=4e-3) # freq_phase

        elif opt.opt_param == "z" or opt.opt_param == "z_pose":
            patch_split = None
            with torch.cuda.amp.autocast():
                generator.get_avg_w()
                if not os.path.exists(load_prev_file):
                    start_iter = 0
                    # initialize z
                    init_z_noise = torch.randn((1, 256), device=device)
                    latent_code = init_z_noise.detach().clone()
                    latent_code.requires_grad = True
                    latent_code = latent_code.to(device)
                else:
                    start_iter = opt.start_iter
                    latents = np.loadtxt(load_prev_file)            
                    latent_code = torch.from_numpy(latents).float().unsqueeze(0).to(device)
                    latent_code.requires_grad = True
                optimizer = torch.optim.Adam([latent_code], lr=1e-1) # z

                    # if opt.opt_param == "z_pose":
                    #     optimizer = torch.optim.Adam([{'params':[latent_code], 'lr':1e-1}, {'params':[h_mean, v_mean], 'lr':4e-3}])
        scaler = torch.cuda.amp.GradScaler()
        scaler._init_scale = 32

        id_loss = IDLoss(face_recog.eval()).cuda()
        save_image(img.detach().cpu(), os.path.join(output_dir, 'input.png'), normalize=True, range=(-1, 1))
        for i in tqdm(range(start_iter, opt.max_iter)):
            loss = 0
            if patch_split is None:
                with torch.cuda.amp.autocast():
                    if opt.opt_param == "freq_phase":
                        freq = latent_code[..., :length]
                        phase = latent_code[..., length:]
                        gen_img = generator.forward_with_frequencies_phase_shifts(freq, phase, **config['camera'])[0]
                        # gen_img = generator.forward_with_frequencies_phase_shifts(freq, phase, **config['camera'], camera_origin=[np.sin(pitch) * np.cos(yaw), np.cos(pitch), np.sin(pitch) * np.sin(yaw)], truncation_psi=opt.psi)[0]

                    elif opt.opt_param == "z":
                        gen_img = generator(latent_code, **config['camera'], truncation_psi=opt.psi)[0]
                        # gen_img = generator(latent_code, **config['camera'], camera_origin=[np.sin(pitch) * np.cos(yaw), np.cos(pitch), np.sin(pitch) * np.sin(yaw)], truncation_psi=opt.psi)[0]

                    img_size = img.size(-1)
                    if opt.lambda_l2 > 0:
                        l2 = torch.mean((gen_img-img)**2) * opt.lambda_l2

                        gen_img_d2 = F.upsample(gen_img, size=(img_size//2,img_size//2), mode='bilinear')
                        img_d2 = F.upsample(img, size=(img_size//2,img_size//2), mode='bilinear')
                        l2 += torch.mean((gen_img_d2-img_d2)**2) * opt.lambda_l2

                        gen_img_d4 = F.upsample(gen_img, size=(img_size//4,img_size//4), mode='bilinear')
                        img_d4 = F.upsample(img, size=(img_size//4,img_size//4), mode='bilinear')
                        l2 += torch.mean((gen_img_d4-img_d4)**2) * opt.lambda_l2
                        l2 = l2 / 3.0

                        loss += l2
                    if opt.lambda_perceptual > 0:
                        if opt.config == 'FFHQ_default' or opt.config == "CelebAHQ_default":
                            gen_features = vgg16(127.5*(gen_img+1), resize_images=False, return_lpips=True)
                            real_features = vgg16(127.5*(img+1), resize_images=False, return_lpips=True)
                            perceptual_loss = ((1000*gen_features-1000*real_features)**2).mean() * opt.lambda_perceptual
                            
                            gen_features_d2 = vgg16(127.5*(gen_img_d2+1), resize_images=False, return_lpips=True)
                            real_features_d2 = vgg16(127.5*(img_d2+1), resize_images=False, return_lpips=True)
                            perceptual_loss += ((1000*gen_features_d2-1000*real_features_d2)**2).mean() * opt.lambda_perceptual
                            
                            gen_features_d4 = vgg16(127.5*(gen_img_d4+1), resize_images=False, return_lpips=True)
                            real_features_d4 = vgg16(127.5*(img_d4+1), resize_images=False, return_lpips=True)
                            perceptual_loss += ((1000*gen_features_d4-1000*real_features_d4)**2).mean() * opt.lambda_perceptual

                            perceptual_loss = perceptual_loss / 3.0
                        elif opt.config == 'CARLA_default' or opt.config == 'CATS_default':
                            perceptual_loss = vgg16(gen_img, img).mean() * opt.lambda_perceptual
                            perceptual_loss += vgg16(gen_img_d2, img_d2).mean() * opt.lambda_perceptual
                            perceptual_loss += vgg16(gen_img_d4, img_d4).mean() * opt.lambda_perceptual
                            perceptual_loss = perceptual_loss / 3.0
                        loss += perceptual_loss
                    if opt.lambda_id > 0:
                        id_l = id_loss(gen_img,img).mean() * opt.lambda_id
                        loss += id_l
                    if opt.opt_param == "freq_phase" and opt.lambda_reg > 0:
                        reg_loss = ((latent_code - torch.cat([avg_freq, avg_phase], dim=-1)) ** 2).mean() *opt.lambda_reg
                        loss += reg_loss
                scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast():
                    gen_img = []
                    if opt.opt_param == "freq_phase":
                        freq = latent_code[..., :length]
                        phase = latent_code[..., length:]
                    with torch.no_grad():
                        for patch_idx in range(patch_split):
                            if opt.opt_param == "freq_phase":
                                gen_imgs_patch = generator.forward_with_frequencies_phase_shifts(freq, phase, **config['camera'], patch=(patch_idx, patch_split))[0]
                            elif opt.opt_param == "z":
                                gen_imgs_patch = generator(latent_code, **config['camera'], truncation_psi=opt.psi, patch=(patch_idx, patch_split))[0]
                            gen_img.append(gen_imgs_patch)
                        gen_img = torch.cat(gen_img,-1).reshape(1,3,generator.img_size,generator.img_size)
                    gen_img.requires_grad = True
                    
                    if opt.lambda_l2 > 0:
                        l2 = torch.mean((gen_img-img)**2) * opt.lambda_l2

                        gen_img_d2 = F.upsample(gen_img, size=(img_size//2,img_size//2), mode='bilinear')
                        img_d2 = F.upsample(img, size=(img_size//2,img_size//2), mode='bilinear')
                        l2 += torch.mean((gen_img_d2-img_d2)**2) * opt.lambda_l2

                        gen_img_d4 = F.upsample(gen_img, size=(img_size//4,img_size//4), mode='bilinear')
                        img_d4 = F.upsample(img, size=(img_size//4,img_size//4), mode='bilinear')
                        l2 += torch.mean((gen_img_d4-img_d4)**2) * opt.lambda_l2
                        l2 = l2 / 3.0

                        loss += l2
                    if opt.lambda_perceptual > 0:
                        if opt.config == 'FFHQ_default' or opt.config == "CelebAHQ_default":
                            gen_features = vgg16(127.5*(gen_img+1), resize_images=False, return_lpips=True)
                            real_features = vgg16(127.5*(img+1), resize_images=False, return_lpips=True)
                            perceptual_loss = ((1000*gen_features-1000*real_features)**2).mean() * opt.lambda_perceptual
                            
                            gen_features_d2 = vgg16(127.5*(gen_img_d2+1), resize_images=False, return_lpips=True)
                            real_features_d2 = vgg16(127.5*(img_d2+1), resize_images=False, return_lpips=True)
                            perceptual_loss += ((1000*gen_features_d2-1000*real_features_d2)**2).mean() * opt.lambda_perceptual
                            
                            gen_features_d4 = vgg16(127.5*(gen_img_d4+1), resize_images=False, return_lpips=True)
                            real_features_d4 = vgg16(127.5*(img_d4+1), resize_images=False, return_lpips=True)
                            perceptual_loss += ((1000*gen_features_d4-1000*real_features_d4)**2).mean() * opt.lambda_perceptual

                            perceptual_loss = perceptual_loss / 3.0
                        elif opt.config == 'CARLA_default' or opt.config == 'CATS_default':
                            perceptual_loss = vgg16(gen_img, img).mean() * opt.lambda_perceptual
                            perceptual_loss += vgg16(gen_img_d2, img_d2).mean() * opt.lambda_perceptual
                            perceptual_loss += vgg16(gen_img_d4, img_d4).mean() * opt.lambda_perceptual
                            perceptual_loss = perceptual_loss / 3.0
                        loss += perceptual_loss
                    if opt.lambda_id > 0:
                        id_l = id_loss(gen_img,img).mean() * opt.lambda_id
                        loss += id_l
                    if opt.opt_param == "freq_phase" and opt.lambda_reg > 0:
                        reg_loss = ((latent_code - torch.cat([avg_freq, avg_phase], dim=-1)) ** 2).mean() *opt.lambda_reg
                        loss += reg_loss

                grad_gen_imgs = torch.autograd.grad(outputs=scaler.scale(loss), inputs=gen_img, create_graph=False)[0]
                grad_gen_imgs = grad_gen_imgs.reshape(1,3,-1)
                grad_gen_imgs = grad_gen_imgs.detach()

                for patch_idx in range(patch_split):
                    with torch.cuda.amp.autocast():
                        if opt.opt_param == "freq_phase":
                            gen_imgs_patch = generator.forward_with_frequencies_phase_shifts(freq, phase, **config['camera'], patch=(patch_idx, patch_split))[0]
                            # gen_imgs_patch = generator.forward_with_frequencies_phase_shifts(freq, phase, **config['camera'], camera_origin=[np.sin(pitch) * np.cos(yaw), np.cos(pitch), np.sin(pitch) * np.sin(yaw)], truncation_psi=opt.psi)[0]
                        elif opt.opt_param == "z":
                            gen_imgs_patch = generator(latent_code, **config['camera'], truncation_psi=opt.psi, patch=(patch_idx, patch_split))[0]
                            # gen_imgs_patch = generator(latent_code, **config['camera'], camera_origin=[np.sin(pitch) * np.cos(yaw), np.cos(pitch), np.sin(pitch) * np.sin(yaw)], truncation_psi=opt.psi)[0]
                    start = generator.img_size*generator.img_size*patch_idx//patch_split
                    end = generator.img_size*generator.img_size*(patch_idx+1)//patch_split
                    gen_imgs_patch.backward(grad_gen_imgs[...,start:end])
                    # scaler.scale(grad_gen_imgs[...,start:end]).backward()

            scaler.unscale_(optimizer)
            # nn.utils.clip_grad_norm_(generator.parameters(), config['optimizer'].get('grad_clip', 0.3))
            nn.utils.clip_grad_norm_(latent_code, config['optimizer'].get('grad_clip', 0.3))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            out_img = gen_img.clone().detach().cpu()
            if i ==0:
                save_image(out_img, os.path.join(output_dir, 'init.png'), normalize=True, range=(-1, 1))

            l_2 = l2.detach().cpu().numpy() if opt.lambda_l2 else 0
            lpips = perceptual_loss.detach().cpu().numpy() if opt.lambda_perceptual else 0
            l_id = id_l.detach().cpu().numpy() if opt.lambda_id else 0
            l_reg = reg_loss.detach().cpu().numpy() if opt.opt_param == "freq_phase" and opt.lambda_reg else 0
            if opt.vis_loss:
                print(f"LPIPS: {lpips}; id_loss: {l_id}; l2: {l_2}; reg: {l_reg};")

            f.write(f"Iter {i}: ")
            f.write(f"LPIPS: {lpips}; id_loss: {l_id}; l2: {l_2}; reg: {l_reg};")
            f.write('\n\n')

            # debug
            # if i == 0:
            #     save_image(out_img, os.path.join(output_dir, '%05d_%s.png'%(i, opt.suffix)), normalize=True, range=(-1, 1))
            #     import ipdb; ipdb.set_trace()

            if i % opt.sv_interval == 0 and i > 0:
                save_image(out_img, os.path.join(output_dir, '%05d_%s.png'%(i, opt.suffix)), normalize=True, range=(-1, 1))
                lat = latent_code.detach().cpu().numpy()
                np.savetxt(os.path.join(output_dir, '%05d_%s.txt' % (i, opt.suffix)), lat)
                if opt.opt_param == "noise_pose":
                    opt_pose = [h_mean.detach().cpu().numpy(), v_mean.detach().cpu().numpy()]
                    np.savetxt(os.path.join(output_dir, '%05d_pose_%s.txt' % (i, opt.suffix)), np.array(opt_pose))


        f.write(f"Save output to {os.path.join(output_dir, '%05d_%s.png' % (i, opt.suffix))}")
        f.close()
        save_image(out_img, os.path.join(output_dir, '%05d_%s.png' %(i, opt.suffix)), normalize=True, range=(-1, 1))
        lat = latent_code.detach().cpu().numpy()
        np.savetxt(os.path.join(output_dir, '%05d_%s.txt' % (i, opt.suffix)), lat)
        if opt.opt_param == "noise_pose":
            opt_pose = [h_mean.detach().cpu().numpy(), v_mean.detach().cpu().numpy()]
            np.savetxt(os.path.join(output_dir, '%05d_pose_%s.txt' % (i, opt.suffix)), np.array(opt_pose))

