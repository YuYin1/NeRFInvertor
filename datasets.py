import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torchvision
import glob
import PIL
import math
import numpy as np
import zipfile
import time
from scipy.io import loadmat

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

def read_latents_txt(name, device="cpu"):
    # load the latent codes for id, expression and so on.

    '''
        the data structure of ffhq_pose
        latents: noise
    '''
    latents = np.loadtxt(name)
    latents = torch.from_numpy(latents).float() #.unsqueeze(0).to(device)

    return latents

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


class CATS(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        for i in range(10):
            try:
                self.data = glob.glob(os.path.join('datasets/cats','*.png'))
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.pose = [os.path.join('datasets/cats/poses',f.split('/')[-1].replace('.png','_pose.npy')) for f in self.data]
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)

        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        flip = (torch.rand(1) < 0.5)
        if flip:
            X = F.hflip(X)
        if self.real_pose:
            P = read_pose_npy(self.pose[index], flip=flip)
        else:
            P = 0

        return X, P

class CATS_finetune(Dataset):
    def __init__(self, opt, img_size, **kwargs):
        super().__init__()
        imgname = opt.target_name

        self.img_size = img_size
        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        for i in range(10):
            try:
                self.data = glob.glob(os.path.join(opt.data_img_dir, f'{imgname}.png'))
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.pose = [os.path.join(opt.data_pose_dir, f.split('/')[-1].replace('.png','_pose.npy')) for f in self.data]
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        self.opt_pose = False
        if opt.data_emd_dir.find('pose') > 0:
            self.opt_pose = True
            self.pose = [os.path.join(opt.data_emd_dir, f'{imgname}/{opt.target_inv_epoch}_pose_.txt')]

        self.emd = [os.path.join(opt.data_emd_dir, f'{imgname}/{opt.target_inv_epoch}_.txt')]
        self.green_bg = opt.green_bg
        self.load_mat = opt.load_mat
        if self.green_bg or self.load_mat:
            self.mat = []
            for img in self.data:
                split = img.split("/")
                self.mat.append(img.replace(split[-1], f"mat256/{split[-1]}"))
            self.transform_mat = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        if self.green_bg:
            mat = PIL.Image.open(self.mat[index])
            # mat.save("mat.png")
            mat_np = np.expand_dims(np.array(mat), axis=2)
            mat_np = mat_np / 255
            X_np = np.array(X)

            # green: [0, 177, 64]
            X_np = (X_np * mat_np + [0, 177, 64] * (1-mat_np)).astype('uint8')
            X = PIL.Image.fromarray(X_np)
            # X.save("rgb_mat.png")

        X = self.transform(X)
        if self.opt_pose:
            P = read_pose_txt(self.pose[index]) # optimized pose
        else:
            P = read_pose_npy(self.pose[index]) # ori pose

        Z = read_latents_txt(self.emd[index])
        if self.load_mat:
            mat = PIL.Image.open(self.mat[index])
            mat = self.transform_mat(mat)

            X = torch.cat((X,mat), 0)

        return X, P, Z

class CARLA(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        for i in range(10):
            try:
                self.data = glob.glob(os.path.join('datasets/carla','*.png'))
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.pose = [os.path.join('datasets/carla/poses',f.split('/')[-1].replace('.png','_extrinsics.npy')) for f in self.data]
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)

        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        flip = (torch.rand(1) < 0.5)
        if flip:
            X = F.hflip(X)
        if self.real_pose:
            P = transform_matrix_to_camera_pos(np.load(self.pose[index]), flip=flip)
        else:
            P = 0

        return X, P


class FFHQ(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        for i in range(10):
            try:
                self.data = glob.glob(os.path.join('datasets/ffhq','*.png'))
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.pose = [os.path.join('datasets/ffhq/poses',f.split('/')[-1].replace('png','mat')) for f in self.data]
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        flip = (torch.rand(1) < 0.5)
        if flip:
            X = F.hflip(X)
        if self.real_pose:
            P = read_pose(self.pose[index],flip=flip)
        else:
            P = 0

        return X, P


class FFHQ_finetune(Dataset):
    def __init__(self, opt, img_size, **kwargs):
        super().__init__()
        imgname = opt.target_name

        self.img_size = img_size
        self.real_pose = False
        if 'real_pose' in kwargs and kwargs['real_pose'] == True:
            self.real_pose = True

        for i in range(10):
            try:
                self.data = glob.glob(os.path.join(opt.data_img_dir, f'{imgname}.png'))
                assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
                if self.real_pose:
                    self.pose = [os.path.join(opt.data_pose_dir, f.split('/')[-1].replace('png','mat')) for f in self.data]
                break
            except:
                print('failed to load dataset, try %02d times'%i)
                time.sleep(0.5)
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        self.opt_pose = False
        if opt.data_emd_dir.find('pose') > 0:
            self.opt_pose = True
            self.pose = [os.path.join(opt.data_emd_dir, f'{imgname}/{opt.target_inv_epoch}_pose_.txt')]

        self.emd = [os.path.join(opt.data_emd_dir, f'{imgname}/{opt.target_inv_epoch}_.txt')]
        self.green_bg = opt.green_bg
        self.load_mat = opt.load_mat
        if self.green_bg or self.load_mat:
            self.mat = []
            for img in self.data:
                split = img.split("/")
                self.mat.append(img.replace(split[-1], f"mat256/{split[-1]}"))
        self.transform_mat = transforms.Compose([transforms.Resize((img_size, img_size), interpolation=1), transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        if self.green_bg:
            mat = PIL.Image.open(self.mat[index])
            # mat.save("mat.png")
            mat_np = np.expand_dims(np.array(mat), axis=2)
            mat_np = mat_np / 255
            X_np = np.array(X)

            # green: [0, 177, 64]
            X_np = (X_np * mat_np + [0, 177, 64] * (1-mat_np)).astype('uint8')
            X = PIL.Image.fromarray(X_np)
            # X.save("rgb_mat.png")

        X = self.transform(X)
        if self.opt_pose:
            P = read_pose_txt(self.pose[index]) # optimized pose
        else:
            P = read_pose(self.pose[index]) # ori pose

        Z = read_latents_txt(self.emd[index])
        if self.load_mat:
            mat = PIL.Image.open(self.mat[index])
            mat = self.transform_mat(mat)

            X = torch.cat((X,mat), 0)

        return X, P, Z


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3

def get_dataset_(dataset, subsample=None, batch_size=1, **kwargs):

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3

def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):

    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        num_workers=1,
        persistent_workers=True,
    )

    return dataloader, 3

def get_dataset_distributed_(_dataset, world_size, rank, batch_size, **kwargs):

    sampler = torch.utils.data.distributed.DistributedSampler(
        _dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        _dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        num_workers=1,
        persistent_workers=True,
    )

    return dataloader, 3


if __name__ == '__main__':
    import imageio
    from tqdm import tqdm
    dataset = FFHQ(64, **{'real_pose': True})
    dataset, _ = get_dataset_(dataset)
    for i, (image, pose) in tqdm(enumerate(dataset)):
        print(pose * 180 / np.pi)
        imageio.imwrite('test.png', ((image.squeeze().permute(1, 2, 0)*0.5+0.5)*255).type(torch.uint8))
        break
