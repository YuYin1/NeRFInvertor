import argparse
import os
import glob
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from tqdm import tqdm
from finetune_loop import training_process

torch.backends.cudnn.benchmark = True

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training") # maximum training epochs
    parser.add_argument('--output_dir', type=str, default='experiments/gram/rendering_results/')
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--load_dir', type=str, default='../pretrained_models')
    parser.add_argument('--data_img_dir', type=str, default='samples/faces/')
    parser.add_argument('--data_pose_dir', type=str, default='samples/faces/poses/')
    parser.add_argument('--data_emd_dir', type=str, default='experiments/gram/inversion')
    parser.add_argument('--pretrain_model', type=str, default='pretrained_models/gram/FFHQ_default/generator.pth')
    parser.add_argument('--config', type=str, default='FACES_default')
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_mesh', type=int, default=1000000)
    parser.add_argument('--port', type=str, default='12356')
    parser.add_argument('--set_step', type=int, default=None) # set to None if train from scratch
    parser.add_argument('--model_save_interval', type=int, default=200)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--log_freq', type=int, default=20)
    parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")  # evaluation interval
    parser.add_argument('--target_inv_epoch', type=str, default='00999', help='epoch num of inversion')
    parser.add_argument('--target_names', type=str, default='')
    parser.add_argument('--load_mask', action='store_true', default=False, help='if specificed, ')

    # loss lambda
    parser.add_argument('--psi', type=float, default=0.7, help='truncation')
    parser.add_argument('--regulizer_alpha', type=float, default=5)

    parser.add_argument('--lambda_loc_reg_l2', type=float, default=1.0)
    parser.add_argument('--lambda_loc_reg_perceptual', type=float, default=1.0)
    parser.add_argument('--lambda_reg_volumeDensity', type=float, default=0)
    parser.add_argument('--lambda_reg_rgbBefAggregation', type=float, default=10)
    parser.add_argument('--lambda_reg_sigmaBefAggregation', type=float, default=0)
    parser.add_argument('--lambda_bg_sigma', type=float, default=10)
    parser.add_argument('--lambda_l2', type=float, default=1)
    parser.add_argument('--lambda_perceptual', type=float, default=1.0)
    parser.add_argument('--lambda_id', type=float, default=0.1)

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

    # parser = facerecon_params(parser)
    opt = parser.parse_args()

    # opt.checkpoints_dir = os.path.join(opt.load_dir, 'FaceRecon_Pytorch/checkpoints')
    # opt.bfm_folder = os.path.join(opt.load_dir, 'FaceRecon_Pytorch/BFM')
    # opt.init_path = os.path.join(opt.load_dir, 'FaceRecon_Pytorch/checkpoints/init_model/resnet50-0676ba61.pth')


    # print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in list(range(torch.cuda.device_count())))
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    print("utilizing %02d gpus"%num_gpus)
    opt.target_names = opt.target_names.split('+')
    output_dir = opt.output_dir
    for target_name in opt.target_names:
        if target_name.find("start_from") >= 0:
            ## start from # in the dataset
            start_ind = int(target_name.split("_")[-1])
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
            opt.output_dir = os.path.join(output_dir, '%s_%s_%s' % (timestr, opt.experiment_name, target_name.split(".")[0]))
            os.makedirs(opt.output_dir, exist_ok=True)

            print("*" * 60)
            print(f"subject: {opt.target_name}")
            print("*" * 60)
            mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
            # try:
            #     mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
            # except:
            #     continue
