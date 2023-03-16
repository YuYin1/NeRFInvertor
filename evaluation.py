import os
import glob
from pickle import FALSE
import shutil
import torch
import numpy as np

from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from pytorch_fid import fid_score
import datasets
from tqdm import tqdm
import copy
import argparse
import imageio
import cv2
import PIL

from FaceRecon_Pytorch.models.arcface_torch.backbones import get_model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# import skimage.metrics as measure
# skimage..peak_signal_noise_ratio(

def calculate_fid(dataset_name, generated_dir, target_size=128):
    real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(target_size))
    for i in range(10):
        try:
            fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], target_size, 'cuda', 2048)
            break
        except:
            print('failed to load evaluation images, try %02d times'%i)

    torch.cuda.empty_cache()

    return fid

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

        loss = F.cosine_similarity(feat_x, feat_y, dim=-1)

        return loss

def convert_rgb_to_y(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)#.detach()

    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
    y_image = image.dot(xform.T) + 16.0

    return y_image


def run_metrics_inversion(opt, dataset, real_images, generated_dir, generated_temp, subfolder):
    img_size = opt.img_size
    transform = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    real_dir = os.path.join(opt.output_dir, dataset + '_real_images_' + str(img_size))
    gen_dir = os.path.join(opt.output_dir, dataset + f'_gen_{generated_temp}_{subfolder}_' + str(img_size))
    sv_file = os.path.join(opt.output_dir, f'metrics_{dataset}_{generated_temp}_{subfolder}_{img_size}.txt')

    if opt.fid:
        # save real images if real_dir not exists
        if not os.path.exists(real_dir):
            os.makedirs(real_dir, exist_ok=True)
            for i, file in enumerate(glob.glob(os.path.join(real_images, "*.png"))):
                if i <= opt.num_real_images:
                    img = PIL.Image.open(file)
                    img = transform(img)
                    save_image(img, os.path.join(real_dir, f'{i:0>5}.png'), normalize=True, range=(-1, 1))
        
        # load generated images
        # import ipdb; ipdb.set_trace()
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir, exist_ok=True)
            subject_count = 0 
            img_count = 0 
            for gen_sub in sorted(glob.glob(os.path.join(generated_dir, f"*{subfolder}"))):
                # subject = int(gen_img.split("/")[-1].split("_")[0])
                if subject_count <= opt.num_subjects:
                    for gen_img in sorted(glob.glob(os.path.join(gen_sub, "*.png"))):
                        img = PIL.Image.open(gen_img)
                        img = transform(img)
                        save_image(img, os.path.join(gen_dir, f'{img_count:0>5}.png'), normalize=True, range=(-1, 1))
                        img_count += 1
                    subject_count += 1

        metrics_dict = calculate_metrics(input1=real_dir, input2=gen_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)
        print("FID:", metrics_dict)
        # shutil.rmtree() 

        with open(sv_file, 'w') as f:
            f.write("FID:\t")
            f.write(str(metrics_dict))
            f.write("\n")


    ## ID
    if opt.id:
        ID_rot=[]
        face_recog = get_model('r50', fp16=False)
        face_recog.load_state_dict(torch.load('../pretrained_models/FaceRecon_Pytorch/models/arcface_torch/backbone.pth', map_location='cpu'))
        print("Load face_recog model for ID loss")
        id_loss = IDLoss(face_recog.eval()).eval().to("cuda")
        subject_count = 0 
        for gen_sub in sorted(glob.glob(os.path.join(generated_dir, f"*{subfolder}"))):
            if subject_count <= opt.num_subjects:
                for gen_img in sorted(glob.glob(os.path.join(gen_sub, "*.png"))):
                    img = PIL.Image.open(gen_img)
                    img = transform(img)
                    subject = gen_img.split("/")[-1].split("_")[0]
                    img_gt = PIL.Image.open(os.path.join(real_images, f"{subject}.png"))
                    img_gt = transform(img_gt)
                    id_l = id_loss(img.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda()).mean()
                    ID_rot.append(float(id_l))
                subject_count += 1

        all_rot_scores = np.array(ID_rot)
        mean = np.mean(all_rot_scores)
        std = np.std(all_rot_scores)
        result_str = 'ID for rotation/expression: {:.4f}+-{:.4f}'.format(mean, std)
        print(result_str)
        with open(sv_file, 'a') as f:
            f.write(result_str)
            f.write("\n")

    # ## PSNR, SSIM, ID
    if opt.rec:
        PSNR_all, SSIM_all, ID_all = [], [], []
        face_recog = get_model('r50', fp16=False)
        face_recog.load_state_dict(torch.load('../pretrained_models/FaceRecon_Pytorch/models/arcface_torch/backbone.pth', map_location='cpu'))
        print("Load face_recog model for ID loss")
        id_loss = IDLoss(face_recog.eval()).eval().to("cuda")
        subject_count = 0 
        for gen_sub in sorted(glob.glob(os.path.join(generated_dir, f"*__"))):
            if subject_count <= opt.num_subjects:
                for gen_img in sorted(glob.glob(os.path.join(gen_sub, "*.png"))):
                    img = PIL.Image.open(gen_img)
                    img = transform(img)
                    subject = gen_img.split("/")[-1].split("_")[0]
                    img_gt = PIL.Image.open(os.path.join(real_images, f"{subject}.png"))
                    img_gt = transform(img_gt)
                    id_l = id_loss(img.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda()).mean()
                    ID_all.append(float(id_l))

                    image1 = convert_rgb_to_y((img+1)/2.0)
                    image2 = convert_rgb_to_y((img_gt+1)/2.0)
                    psnr = peak_signal_noise_ratio(image1, image2, data_range=1)
                    ssim = structural_similarity(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
                                        sigma=1.5, data_range=1)
                    PSNR_all.append(psnr)
                    SSIM_all.append(ssim)
                subject_count += 1
        all_scores = np.array(ID_all)
        mean = np.mean(all_scores)
        std = np.std(all_scores)
        result_str = "Rreconstruction: \n"
        result_str_psnr = '\tPSNR: {:.4f} \n'.format(np.mean(PSNR_all))
        result_str_ssim = '\tSSIM: {:.4f} \n'.format(np.mean(SSIM_all))
        result_str_id = '\tID score: {:.4f}+-{:.4f}'.format(mean, std)
        print(result_str)
        print(result_str_psnr)
        print(result_str_ssim)
        print(result_str_id)

        with open(sv_file, 'a') as f:
            f.write(result_str)
            f.write(result_str_psnr)
            f.write(result_str_ssim)
            f.write(result_str_id)
            f.write("\n")


def run_metrics_finetune(opt, dataset, real_images, generated_dir, generated_temp, subfolder, method):
    img_size = opt.img_size
    transform = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    real_dir = os.path.join(opt.output_dir, dataset + '_real_images_' + str(img_size))
    gen_dir = os.path.join(opt.output_dir, dataset + f'_gen_{generated_temp}_{method}_{subfolder}_' + str(img_size))
    sv_file = os.path.join(opt.output_dir, f'metrics_{dataset}_{generated_temp}_{method}_{subfolder}_{img_size}.txt')

    if opt.fid:
        # save real images if real_dir not exists
        if not os.path.exists(real_dir):
            os.makedirs(real_dir, exist_ok=True)
            for i, file in enumerate(glob.glob(os.path.join(real_images, "*.png"))):
                if i <= opt.num_real_images:
                    img = PIL.Image.open(file)
                    img = transform(img)
                    save_image(img, os.path.join(real_dir, f'{i:0>5}.png'), normalize=True, range=(-1, 1))
        
        # load generated images
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir, exist_ok=True)
            subject_count = 0 
            img_count = 0 
            for gen_sub in sorted(glob.glob(os.path.join(generated_dir, method, f"*_{method}_*", f"*{subfolder}"))):
                subject = int(gen_sub.split("/")[-1].split("_")[0])
                if subject_count <= opt.num_subjects and subject <= 10961:
                    for gen_img in sorted(glob.glob(os.path.join(gen_sub, "*.png"))):
                        img = PIL.Image.open(gen_img)
                        img = transform(img)
                        save_image(img, os.path.join(gen_dir, f'{img_count:0>5}.png'), normalize=True, range=(-1, 1))
                        img_count += 1
                    subject_count += 1

        metrics_dict = calculate_metrics(input1=real_dir, input2=gen_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)
        print("FID: ", metrics_dict)
        # shutil.rmtree() 

        with open(sv_file, 'w') as f:
            f.write("FID:\t")
            f.write(str(metrics_dict))
            f.write("\n")

    ## ID
    if opt.id:
        ID_rot=[]
        face_recog = get_model('r50', fp16=False)
        face_recog.load_state_dict(torch.load('../pretrained_models/FaceRecon_Pytorch/models/arcface_torch/backbone.pth', map_location='cpu'))
        print("Load face_recog model for ID loss")
        id_loss = IDLoss(face_recog.eval()).eval().to("cuda")
        subject_count = 0 
        for gen_sub in sorted(glob.glob(os.path.join(generated_dir, method, f"*_{method}_*", f"*{subfolder}"))):
            subject = int(gen_sub.split("/")[-1].split("_")[0])
            if subject_count <= opt.num_subjects and subject <= 10961:
                for gen_img in sorted(glob.glob(os.path.join(gen_sub, "*.png"))):
                    img = PIL.Image.open(gen_img)
                    img = transform(img)
                    subject = gen_img.split("/")[-1].split("_")[0]
                    img_gt = PIL.Image.open(os.path.join(real_images, f"{subject}.png"))
                    img_gt = transform(img_gt)
                    id_l = id_loss(img.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda()).mean()
                    ID_rot.append(float(id_l))
                subject_count += 1
        all_rot_scores = np.array(ID_rot)
        mean = np.mean(all_rot_scores)
        std = np.std(all_rot_scores)
        result_str = 'ID for rotation/expression: {:.4f}+-{:.4f}'.format(mean, std)
        print(result_str)
        with open(sv_file, 'a') as f:
            f.write(result_str)
            f.write("\n")

    # ## PSNR, SSIM, ID
    if opt.rec:
        PSNR_all, SSIM_all, ID_all = [], [], []
        face_recog = get_model('r50', fp16=False)
        face_recog.load_state_dict(torch.load('../pretrained_models/FaceRecon_Pytorch/models/arcface_torch/backbone.pth', map_location='cpu'))
        print("Load face_recog model for ID loss")
        id_loss = IDLoss(face_recog.eval()).eval().to("cuda")
        subject_count = 0 
        for gen_sub in sorted(glob.glob(os.path.join(generated_dir, method, f"*_{method}_*", f"*__"))):
            subject = int(gen_sub.split("/")[-1].split("_")[0])
            if subject_count <= opt.num_subjects and subject <= 10961:
                for gen_img in sorted(glob.glob(os.path.join(gen_sub, "*.png"))):
                    img = PIL.Image.open(gen_img)
                    img = transform(img)
                    subject = gen_img.split("/")[-1].split("_")[0]
                    img_gt = PIL.Image.open(os.path.join(real_images, f"{subject}.png"))
                    img_gt = transform(img_gt)
                    id_l = id_loss(img.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda()).mean()
                    ID_all.append(float(id_l))

                    image1 = convert_rgb_to_y((img+1)/2.0)
                    image2 = convert_rgb_to_y((img_gt+1)/2.0)
                    psnr = peak_signal_noise_ratio(image1, image2, data_range=1)
                    ssim = structural_similarity(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
                                        sigma=1.5, data_range=1)
                    PSNR_all.append(psnr)
                    SSIM_all.append(ssim)
                subject_count += 1
        all_scores = np.array(ID_all)
        mean = np.mean(all_scores)
        std = np.std(all_scores)
        result_str = "Rreconstruction: \n"
        result_str_psnr = '\tPSNR: {:.4f} \n'.format(np.mean(PSNR_all))
        result_str_ssim = '\tSSIM: {:.4f} \n'.format(np.mean(SSIM_all))
        result_str_id = '\tID score: {:.4f}+-{:.4f}'.format(mean, std)
        print(result_str)
        print(result_str_psnr)
        print(result_str_ssim)
        print(result_str_id)

        with open(sv_file, 'a') as f:
            f.write(result_str)
            f.write(result_str_psnr)
            f.write(result_str_ssim)
            f.write(result_str_id)
            f.write("\n")


# python evaluation.py --fid --id --rec --dataset CelebAHQ or FFHQ
# rm -r ../exp_gram/evaluation/CelebAHQ_gen_edit_inverted_images_celebahq_*
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='../exp_gram/evaluation/')
    parser.add_argument('--dataset', type=str, default='CelebAHQ')
    parser.add_argument('--num_real_images', type=int, default=2000)
    parser.add_argument('--num_subjects', type=int, default=150)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--fid', action='store_true', default=False)
    parser.add_argument('--id', action='store_true', default=False)
    parser.add_argument('--rec', action='store_true', default=False)
    opt = parser.parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)
    
    dataset = opt.dataset
    if dataset == "CelebAHQ":
        real_images = "../../Dataset/NeRFGAN/celebahq_test256_align_new_mirror_wo_t"

        ## W inversion
        generated_dir = "../exp_gram/edit_inversed/celebahq_FreqPhase"
        generated_temp = "celebahq_FreqPhase"
        subfolders = ["_rotate_"]
        for subfolder in subfolders:
            run_metrics_inversion(opt, dataset, real_images, generated_dir, generated_temp, subfolder)

        ## Z inversion
        generated_dir = "../exp_gram/edit_inversed/celebahq_Z"
        generated_temp = "celebahq_Z"
        subfolders = ["_rotate_"]
        for subfolder in subfolders:
            run_metrics_inversion(opt, dataset, real_images, generated_dir, generated_temp, subfolder)

        ## finetune
        generated_dir = "../exp_gram/edit_after_finetune/celebahq"
        generated_temp = "edit_after_finetune"
        methods = ["woreg", "noise2dloss", "noise3dloss", "noise3dlossWMasks"]
        subfolders = ["_rotate_"]
        for method in methods:
            print("-"*30, f"  {methods}  ","-"*30)
            for subfolder in subfolders:
                print("#"*10, f"  {subfolder}  ","#"*10)
                run_metrics_finetune(opt, dataset, real_images, generated_dir, generated_temp, subfolder, method)
    elif dataset == "FFHQ":
        real_images = "../../Dataset/NeRFGAN/image256_align_new_mirror_wo_t"

        # W inversion
        generated_dir = "../exp_gram/edit_inversed/ffhq_FreqPhase"
        generated_temp = "ffhq_FreqPhase"
        subfolders = ["_rotate_"]
        for subfolder in subfolders:
            run_metrics_inversion(opt, dataset, real_images, generated_dir, generated_temp, subfolder)

        ## Z inversion
        generated_dir = "../exp_gram/edit_inversed/ffhq_Z"
        generated_temp = "ffhq_Z"
        subfolders = ["_rotate_"]
        for subfolder in subfolders:
            run_metrics_inversion(opt, dataset, real_images, generated_dir, generated_temp, subfolder)

        ## finetune
        generated_dir = "../exp_gram/edit_after_finetune/ffhq"
        generated_temp = "edit_after_finetune"
        methods = ["woreg", "noise2dloss", "noise3dloss", "noise3dlossWMasks"]
        subfolders = ["_rotate_"]
        for method in methods:
            print("-"*30, f"  {methods}  ","-"*30)
            for subfolder in subfolders:
                print("#"*10, f"  {subfolder}  ","#"*10)
                run_metrics_finetune(opt, dataset, real_images, generated_dir, generated_temp, subfolder, method)


