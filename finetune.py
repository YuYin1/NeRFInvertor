import argparse
import os
import glob
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from tqdm import tqdm
# from finetune import training_process
from finetune_loop import training_process
import FaceRecon_Pytorch.util.util as util

torch.backends.cudnn.benchmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in list(range(torch.cuda.device_count())))

def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    torch.cuda.set_device(rank)
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)

    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    synchronize()

def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, opt):
    torch.manual_seed(0)

    setup(rank, world_size, opt.port)   # multi_process initialization
    device = torch.device(rank)
    training_process(rank, world_size, opt, device) # main training loop
    cleanup()

def facerecon_params(parser):
    """Define the common options that are used in both training and test."""
    # basic parameters
    parser.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./FaceRecon_Pytorch/checkpoints', help='models are saved here')
    parser.add_argument('--vis_batch_nums', type=float, default=1, help='batch nums of images for visulization')
    parser.add_argument('--eval_batch_nums', type=float, default=float('inf'), help='batch nums of images for evaluation')
    parser.add_argument('--use_ddp', type=util.str2bool, nargs='?', const=False, default=False, help='whether use distributed data parallel')
    parser.add_argument('--ddp_port', type=str, default='12356', help='ddp port')
    parser.add_argument('--display_per_batch', type=util.str2bool, nargs='?', const=True, default=True, help='whether use batch to show losses')
    parser.add_argument('--add_image', type=util.str2bool, nargs='?', const=True, default=True, help='whether add image to tensorboard')
    parser.add_argument('--world_size', type=int, default=1, help='batch nums of images for evaluation')

    # model parameters
    parser.add_argument('--model', type=str, default='facerecon', help='chooses which model to use.')

    # additional parameters
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

    # self.initialized = True
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
    parser.add_argument('--init_path', type=str, default='./FaceRecon_Pytorch/checkpoints/init_model/resnet50-0676ba61.pth')
    parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./FaceRecon_Pytorch/BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)
    parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
        )
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training") # maximum training epochs
    parser.add_argument('--output_dir', type=str, default='../experiments/trained/')
    parser.add_argument('--experiment_name', type=str, default='finetune_noisepose3dloss')
    parser.add_argument('--load_dir', type=str, default='../pretrained_models')
    parser.add_argument('--data_img_dir', type=str, default='../../Dataset/NeRFGAN/image256_align_new_mirror_wo_t')
    parser.add_argument('--data_pose_dir', type=str, default='../../Dataset/NeRFGAN/ffhq_pose_align_new_mirror')
    parser.add_argument('--data_emd_dir', type=str, default='../experiments/inversion_ffhq_noise/')
    parser.add_argument('--pretrain_model', type=str, default='AniFaceGAN_results_paper/20220514-122632_warm_up_deform_2000_switch_interval_3_DIF_lambda_0_ths_0.000010')
    parser.add_argument('--config', type=str, default='FFHQ_default')
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_mesh', type=int, default=1000000)
    parser.add_argument('--port', type=str, default='12356')
    parser.add_argument('--set_step', type=int, default=None) # set to None if train from scratch
    parser.add_argument('--model_save_interval', type=int, default=200)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")  # evaluation interval
    parser.add_argument('--target_inv_epoch', type=str, default='01999', help='epoch num of inversion')
    parser.add_argument('--target_names', type=str, default='')
    parser.add_argument('--load_mat', action='store_true', default=False, help='if specificed, ')
    parser.add_argument('--green_bg', action='store_true', default=False, help='if specificed, ')

    # loss lambda
    parser.add_argument('--psi', type=float, default=0.7, help='truncation')
    parser.add_argument('--loc_reg', action='store_true', help='if specificed, ')
    parser.add_argument('--regulizer_alpha', type=float, default=5)
    parser.add_argument('--lambda_loc_reg_l2', type=float, default=1.0)
    parser.add_argument('--lambda_loc_reg_perceptual', type=float, default=1.0)
    parser.add_argument('--lambda_reg_volumeDensity', type=float, default=0)
    parser.add_argument('--lambda_reg_rgbBefAggregation', type=float, default=10)
    parser.add_argument('--lambda_reg_sigmaBefAggregation', type=float, default=0)
    parser.add_argument('--lambda_bg_sigma', type=float, default=0)
    parser.add_argument('--lambda_sparse', type=float, default=0)
    parser.add_argument('--lambda_l2', type=float, default=1)
    parser.add_argument('--lambda_perceptual', type=float, default=1.0)
    parser.add_argument('--lambda_id', type=float, default=0.1)
    # parser.add_argument('--gpus', type=str, default='0')

    parser.add_argument('--warm_up_deform', type=int, default=2000, help='the warm up iterations for training DIF solely')
    parser.add_argument('--switch_interval', type=int, default=3, help='switch inverval between deform net and GRAM, 3 means training deformnet twice and train GRAM once')
    parser.add_argument('--gen_gt', action='store_true', help='gen_gt means for BFM, samples points on the rays; otherwise directly use points from BFM for training')
    parser.add_argument('--with_smoothness', action='store_true', help='whether use smoothness, need a high memory demand')
    parser.add_argument('--debug_mode', action='store_true', help='if specificed, use the debug mode')
    parser.add_argument('--real_latents', action='store_true', help='if specificed, use the real latents')
    parser.add_argument('--gen_points_threshold', type=float, default=0.00001)
    parser.add_argument('--sample_rays', action='store_true', help='whether sample rays during the training of DIFNET')
    parser.add_argument('--train_rignerf', action='store_true', help='whether use rignerf methods to train 3dmm guidance')
    parser.add_argument('--sample_3dmm', type=float, default=0.5, help='sample how much points on 3DMM face')
    parser.add_argument('--generator_model', type=str, default='GRAM', help='the generative model, choose from GRAM or pi-gan')
    parser.add_argument('--neutral_ratio', type=float, default=0.1, help='the ratio of input to simulate canonic process')
    parser.add_argument('--n_workers', type=int, default=1, help='the workers for dataloader')
    parser.add_argument('--deform_backbone', type=str, default='siren', help='the backbone of siren')
    
    parser.add_argument('--to_gram', type=str, default='v1', help='the backbone of siren')

    parser = facerecon_params(parser)
    opt = parser.parse_args()

    opt.checkpoints_dir = os.path.join(opt.load_dir, 'FaceRecon_Pytorch/checkpoints')
    opt.bfm_folder = os.path.join(opt.load_dir, 'FaceRecon_Pytorch/BFM')
    opt.init_path = os.path.join(opt.load_dir, 'FaceRecon_Pytorch/checkpoints/init_model/resnet50-0676ba61.pth')


    # print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in list(range(torch.cuda.device_count())))
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    # num_gpus = len(opt.gpus.split(','))

    print("utilizing %02d gpus"%num_gpus)
    opt.target_names = opt.target_names.split('+')
    output_dir = opt.output_dir
    for target_name in opt.target_names:
        if target_name.find("start_from") >= 0:
            ## start from # in the dataset
            start_ind = int(target_name.split("_")[-1])
            # img_paths_all = sorted(glob.glob(opt.data_emd_dir))
            img_paths_all = sorted(os.listdir(opt.data_emd_dir))
            for i, file in enumerate(img_paths_all):
                if i < start_ind:
                    continue
                # -------------- modify the output dir
                opt.target_name = file
                timestr = time.strftime("%Y%m%d-%H%M%S")
                opt.output_dir = os.path.join(output_dir, '%s_%s_%s' % (timestr, opt.experiment_name, file))
                os.makedirs(opt.output_dir, exist_ok=True)

                print("*" * 60)
                print(f"subject: {opt.target_name} (idx{i})")
                print("*" * 60)
                mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
        else:
            ## use specific target_names
            # -------------- modify the output dir
            opt.target_name = target_name
            timestr = time.strftime("%Y%m%d-%H%M%S")
            opt.output_dir = os.path.join(output_dir, '%s_%s_%s' % (timestr, opt.experiment_name, target_name))
            os.makedirs(opt.output_dir, exist_ok=True)

            print("*" * 60)
            print(f"subject: {opt.target_name}")
            print("*" * 60)
            mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
            # try:
            #     mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
            # except:
            #     continue
