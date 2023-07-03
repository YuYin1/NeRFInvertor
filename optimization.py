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
import torch.nn.functional as F
import torch.nn as nn
import importlib
from torch_ema import ExponentialMovingAverage
from utils.arcface import get_model


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
    if opt.config == 'FACES_default':
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)
    elif opt.config == 'CATS_default':   # CATS, CARLA
        import lpips
        vgg16 = lpips.LPIPS(net='vgg').eval().to(device) # closer to "traditional" perceptual loss, when used for optimization
    else:
        raise

    face_recog = get_model('r50', fp16=False)
    face_recog.load_state_dict(torch.load('pretrained_models/arcface.pth'))
    face_recog.eval()

    return generator, vgg16, face_recog

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_file', type=str, default='pretrained_models/gram/FACES_default/generator.pth')
    parser.add_argument('--output_dir', type=str, default='experiments/gram/inversion')
    parser.add_argument('--data_img_dir', type=str, default='samples/faces/')
    parser.add_argument('--data_pose_dir', type=str, default='samples/faces/camerapose/')
    parser.add_argument('--name', type=str, default=None, help="specifc image name (e.g. '28606.png'), or None (will invert all images)")
    parser.add_argument('--config', type=str, default='FACES_default')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument('--num_images', type=int, default=-1, help="the number of inverted images or -1 (all images)")
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
    parser.add_argument('--psi', type=str, default=0.7)
    parser.add_argument('--lambda_perceptual', type=float, default=1)
    parser.add_argument('--lambda_l2', type=float, default=0.01)
    parser.add_argument('--lambda_id', type=float, default=0.01)
    parser.add_argument('--lambda_reg', type=float, default=0.04)

    parser.add_argument('--start_iter', type=int, default=2000)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--sv_interval', type=int, default=50)
    parser.add_argument('--vis_loss', type=bool, default=False)

    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = getattr(configs, opt.config)

    ## load models
    generator, vgg16, face_recog = load_models(opt, config, device)
    generator.renderer.lock_view_dependence = True

    ## load data
    img_size = config['global']['img_size']
    transform = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # search all images
    img_fullpaths_all = []
    if opt.name:
        name = opt.name
        img_fullpath = os.path.join(opt.data_img_dir, f"{opt.name}.png")
        img_fullpaths_all.append(img_fullpath)
    else:
        img_fullpaths_all = sorted(glob.glob(os.path.join(opt.data_img_dir, f"*.png")))
        img_fullpaths_all = img_fullpaths_all[:opt.num_images]
    img_fullpaths = []
    for imgpath in img_fullpaths_all:
        subject = imgpath.split('/')[-1].split('.')[0]
        inv_path = os.path.join(opt.output_dir, subject, f"{(opt.max_iter-1):05d}_.txt")
        if not os.path.exists(inv_path):
            img_fullpaths.append(imgpath)
        else:
            print(f"Ignoring {subject}...")


    ## start optimization
    for img_fullpath in img_fullpaths:
        # load image and mat file
        print(f"Processing {img_fullpath}...")
        img = Image.open(img_fullpath)
        img = transform(img).cuda()
        img = img.unsqueeze(0)
        name = img_fullpath.split("/")[-1][:-4]
        if opt.config.find('FACES') >= 0:
            mat_fullpath = os.path.join(opt.data_pose_dir, f"{name.split('.')[0]}.mat")
            pose = read_pose(mat_fullpath)
        elif opt.config.find('CATS') >= 0:   # CATS
            mat_fullpath = os.path.join(opt.data_pose_dir, f"{name.split('.')[0]}_pose.npy")
            pose = read_pose_npy(mat_fullpath)
        else:
            raise
        # load camera pose
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


        scaler = torch.cuda.amp.GradScaler()
        scaler._init_scale = 32

        id_loss = IDLoss(face_recog.eval()).cuda()
        save_image(img.detach().cpu(), os.path.join(output_dir, 'input.png'), normalize=True, range=(-1, 1))
        for i in tqdm(range(start_iter, opt.max_iter)):
            loss = 0
            if patch_split is None:
                with torch.cuda.amp.autocast():
                    gen_img = generator(latent_code, **config['camera'], truncation_psi=opt.psi)[0]

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
                        if opt.config == 'FACES_default':
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
                        elif opt.config == 'CATS_default':
                            perceptual_loss = vgg16(gen_img, img).mean() * opt.lambda_perceptual
                            perceptual_loss += vgg16(gen_img_d2, img_d2).mean() * opt.lambda_perceptual
                            perceptual_loss += vgg16(gen_img_d4, img_d4).mean() * opt.lambda_perceptual
                            perceptual_loss = perceptual_loss / 3.0
                        loss += perceptual_loss
                    if opt.lambda_id > 0:
                        id_l = id_loss(gen_img,img).mean() * opt.lambda_id
                        loss += id_l
                scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast():
                    gen_img = []
                    with torch.no_grad():
                        for patch_idx in range(patch_split):
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
                        if opt.config == 'FACES_default':
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
                        elif opt.config == 'CATS_default':
                            perceptual_loss = vgg16(gen_img, img).mean() * opt.lambda_perceptual
                            perceptual_loss += vgg16(gen_img_d2, img_d2).mean() * opt.lambda_perceptual
                            perceptual_loss += vgg16(gen_img_d4, img_d4).mean() * opt.lambda_perceptual
                            perceptual_loss = perceptual_loss / 3.0
                        loss += perceptual_loss
                    if opt.lambda_id > 0:
                        id_l = id_loss(gen_img,img).mean() * opt.lambda_id
                        loss += id_l

                grad_gen_imgs = torch.autograd.grad(outputs=scaler.scale(loss), inputs=gen_img, create_graph=False)[0]
                grad_gen_imgs = grad_gen_imgs.reshape(1,3,-1)
                grad_gen_imgs = grad_gen_imgs.detach()

                for patch_idx in range(patch_split):
                    with torch.cuda.amp.autocast():
                        gen_imgs_patch = generator(latent_code, **config['camera'], truncation_psi=opt.psi, patch=(patch_idx, patch_split))[0]

                    start = generator.img_size*generator.img_size*patch_idx//patch_split
                    end = generator.img_size*generator.img_size*(patch_idx+1)//patch_split
                    gen_imgs_patch.backward(grad_gen_imgs[...,start:end])


            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(latent_code, config['optimizer'].get('grad_clip', 0.3))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            out_img = gen_img.clone().detach().cpu()
            if i ==0:
                save_image(out_img, os.path.join(output_dir, 'init.png'), normalize=True, range=(-1, 1))

            l_2 = l2.detach().cpu().numpy() if opt.lambda_l2 else 0
            lpips = perceptual_loss.detach().cpu().numpy() if opt.lambda_perceptual else 0
            l_id = id_l.detach().cpu().numpy() if opt.lambda_id else 0

            if opt.vis_loss:
                print(f"LPIPS: {lpips}; id_loss: {l_id}; l2: {l_2};")

            f.write(f"Iter {i}: ")
            f.write(f"LPIPS: {lpips}; id_loss: {l_id}; l2: {l_2};")
            f.write('\n\n')

            # debug
            # if i == 0:
            #     save_image(out_img, os.path.join(output_dir, '%05d_%s.png'%(i, opt.suffix)), normalize=True, range=(-1, 1))
            #     import ipdb; ipdb.set_trace()

            if i % opt.sv_interval == 0 and i > 0:
                save_image(out_img, os.path.join(output_dir, '%05d_%s.png'%(i, opt.suffix)), normalize=True, range=(-1, 1))
                lat = latent_code.detach().cpu().numpy()
                np.savetxt(os.path.join(output_dir, '%05d_%s.txt' % (i, opt.suffix)), lat)

        f.write(f"Save output to {os.path.join(output_dir, '%05d_%s.png' % (i, opt.suffix))}")
        f.close()
        save_image(out_img, os.path.join(output_dir, '%05d_%s.png' %(i, opt.suffix)), normalize=True, range=(-1, 1))
        lat = latent_code.detach().cpu().numpy()
        np.savetxt(os.path.join(output_dir, '%05d_%s.txt' % (i, opt.suffix)), lat)

