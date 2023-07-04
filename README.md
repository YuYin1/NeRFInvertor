# NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-shot Real Image Animation, CVPR'23
<p align="center"> 
<img src="/docs/teaser.mov">
</p>

This is an official pytorch implementation of our NeRFInvertor paper:

Y. Yin, K. Ghasedi, H. Wu, J. Yang, X. Tong, Y. Fu, **NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-shot Real Image Animation**, IEEE Computer Vision and Pattern Recognition (CVPR), 2023.


###[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yin_NeRFInvertor_High_Fidelity_NeRF-GAN_Inversion_for_Single-Shot_Real_Image_Animation_CVPR_2023_paper.pdf)] [[ArXiv](https://arxiv.org/abs/2211.17235)] [[Project Page](https://yuyin1.github.io/NeRFInvertor_Homepage/)] ### 

Abstract: _Nerf-based Generative models (NeRF-GANs) have shown impressive capacity in generating high-quality images with consistent 3D geometry. In this paper, we propose a universal method to surgically fine-tune these NeRF-GANs in order to achieve high-fidelity animation of real subjects only by a single image. Given the optimized latent code for an out-of-domain real image, we employ 2D loss functions on the rendered image to reduce the identity gap. Furthermore, our method leverages explicit and implicit 3D regularizations using the in-domain neighborhood samples around the optimized latent code to remove geometrical and visual artifacts._


## Recent Updates
**2023.06.01:** Inversion of [GRAM](https://github.com/microsoft/GRAM/)

**TODO:**
- Inversion of [EG3D](https://github.com/NVlabs/eg3d)
- Inversion of [AnifaceGAN](https://yuewuhkust.github.io/AniFaceGAN/)

## Requirements
- Currently only Linux is supported.
- 64-bit Python 3.8 installation or newer. We recommend using Anaconda3.
- One or more high-end NVIDIA GPUs, NVIDIA drivers, and CUDA toolkit 10.1 or newer. We recommend using Tesla V100 GPUs with 32 GB memory for training to reproduce the results in the paper. 

## Installation
Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/YuYin1/NeRFInvertor.git
cd NeRFInvertor
conda env create -f environment.yml
source activate nerfinvertor
```

## Pre-trained models
We provide various auxiliary models needed for NeRF-GAN inversion task. This includes the NeRF-based generators and pre-trained models used for loss computation.
# Pretrained NeRF-GANs
|Model|Dataset|Resolution|Download|
|:----:|:----:|:-------:|:-----------:|
| GRAM | FFHQ | 256x256 | [Github link](https://github.com/microsoft/GRAM/tree/main/pretrained_models/FFHQ_default) |
| GRAM | Cats | 256x256 | [Github link](https://github.com/microsoft/GRAM/tree/main/pretrained_models/CATS_default) |
| EG3D | FFHQ | 256x256 | [Github link](https://github.com/NVlabs/eg3d/blob/main/docs/models.md) |
| AnifaceGAN | FFHQ | 512x512 | [Github link](https://yuewuhkust.github.io/AniFaceGAN/) |
<!-- |      | CARLA| 128x128 | [Github link](https://github.com/microsoft/GRAM/tree/main/pretrained_models/CARLA_default)| -->

## Prepare Dataset
- Sample dataset
- FFHQ: Download the [original 1024x1024 images](https://github.com/NVlabs/ffhq-dataset). We additionally provide [detected 5 facial landmarks (google drive)](https://drive.google.com/file/d/1bOefjWzNGzjJ65J5WT9V0QrsrNhKjjCb/view?usp=sharing) for image preprocessing and [face poses (google drive)](https://drive.google.com/file/d/1kb-PeNhOEmN1Gs8e0xF3aLjsjHe01sVb/view?usp=sharing) estimated by [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch) for training. Download all files and organize them as follows:
```
GRAM/
│
└─── raw_data/
    |
    └─── ffhq/
	│
	└─── *.png   # original 1024x1024 images
	│
        └─── lm5p/   # detected 5 facial landmarks
	|   |
        |   └─── *.txt
	|
	└─── poses/  # estimated face poses
	    |
	    └─── *.mat    
```
- Cats: Download the original cat images and provided landmarks using this [link](https://archive.org/details/CAT_DATASET) and organize all files as follows:
```
GRAM/
│
└─── raw_data/
    |
    └─── cats/
	│
	└─── *.jpg   # original images
	│
        └─── *.jpg.cat   # provided landmarks
```

## Inversion
# Optimize latent codes
In order to invert a real image and edit it you should first align and crop it to the correct size. 
Use --name=image_name.png to invert a specific image, otherwise, the following commond will invert all images in img_dir 
```
python optimization.py \
    --generator_file='pretrained_models/gram/FFHQ_default/generator.pth' \
    --output_dir='experiments/gram/optimization' \
    --data_img_dir='samples/faces/' \
    --data_pose_dir='samples/faces/camerapose/' \
    --config='FACES_default' \
    --max_iter=1000
```

# Finetune NeRFGANs
```
CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
    --target_names='R1.png+R2.png' \
    --config='FACES_finetune' \
    --output_dir='experiments/gram/finetuned_model/' \
    --data_img_dir='samples/faces/' \
    --data_pose_dir='samples/faces/camerapose/'  \
    --data_emd_dir='experiments/gram/optimization/' \
    --pretrain_model='pretrained_models/gram/FFHQ_default/generator.pth' \
    --load_mask \
    --regulizer_alpha=5 \
    --lambda_id=0.1 \
    --lambda_reg_rgbBefAggregation 10 \
    --lambda_bg_sigma 10
```

## Inference
# Rendering results for finetuned models
```
CUDA_VISIBLE_DEVICES=0 python rendering_using_finetuned_model.py \
    --generator_file='experiments/gram/finetuned_model/000990/generator.pth' \
    --target_name='000990' \
    --output_dir='experiments/gram/rendering_results/' \
    --data_img_dir='samples/faces/' \
    --data_pose_dir='samples/faces/camerapose/'  \
    --data_emd_dir='experiments/gram/optimization/' \
    --config='FACES_finetune' \
    --image_size 256 \
    --gen_video
```
<!-- Rendering results for Cats -->
<!-- CUDA_VISIBLE_DEVICES=0 python rendering_using_finetuned_model.py \
    --generator_file='experiments/gram/finetuned_model/00000005_001/generator.pth' \
    --target_name='00000005_001' \
    --output_dir='experiments/gram/rendering_results/' \
    --data_img_dir='samples/cats/' \
    --data_pose_dir='samples/cats/camerapose/'  \
    --data_emd_dir='experiments/gram/optimization/' \
    --config='CATS_finetune' \
    --image_size 256 \
    --gen_video -->


## Citation
	@inproceedings{yin2023nerfinvertor,
	  title={NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-shot Real Image Animation},
	  author={Yin, Yu and Ghasedi, Kamran and Wu, HsiangTao and Yang, Jiaolong and Tong, Xin and Fu, Yun},
	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	  pages={8539--8548},
	  year={2023}
	}