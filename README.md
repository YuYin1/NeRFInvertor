# NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-shot Real Image Animation, CVPR'23
<p align="center"> 
<img src="/doc/teaser.mov">
</p>

This is an official pytorch implementation of the following paper:

Y. Yin, K. Ghasedi, H. Wu, J. Yang, X. Tong, Y. Fu, **NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-shot Real Image Animation**, IEEE Computer Vision and Pattern Recognition (CVPR), 2023.


### [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yin_NeRFInvertor_High_Fidelity_NeRF-GAN_Inversion_for_Single-Shot_Real_Image_Animation_CVPR_2023_paper.pdf)]
[[ArXiv](https://arxiv.org/abs/2211.17235)]
[[Project Page](https://yuyin1.github.io/NeRFInvertor_Homepage/)]### 

Abstract: _Nerf-based Generative models have shown impressive capacity in generating high-quality images with consistent 3D geometry. Despite successful synthesis of fake identity images randomly sampled from latent space, adopting these models for generating face images of real subjects is still a challenging task due to its so-called inversion issue. In this paper, we propose a universal method to surgically fine-tune these NeRF-GAN models in order to achieve high-fidelity animation of real subjects only by a single image. Given the optimized latent code for an out-of-domain real image, we employ 2D loss functions on the rendered image to reduce the identity gap. Furthermore, our method leverages explicit and implicit 3D regularizations using the in-domain neighborhood samples around the optimized latent code to remove geometrical and visual artifacts. Our experiments confirm the effectiveness of our method in realistic, high-fidelity, and 3D consistent animation of real faces on multiple NeRF-GAN models across different datasets._

## Requirements
- Currently only Linux is supported.
- 64-bit Python 3.6 installation or newer. We recommend using Anaconda3.
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
Checkpoints for pre-trained models used in our paper (default settings) are as follows.
|Model|Dataset|Resolution|Download|
|:----:|:-----------:|:-----------:|:-----------:|
|      | FFHQ  | 256x256 | [Github link](https://github.com/microsoft/GRAM/tree/main/pretrained_models/FFHQ_default) |
| GRAM | Cats  | 256x256 | [Github link](https://github.com/microsoft/GRAM/tree/main/pretrained_models/CATS_default) |
|      | CARLA | 128x128 | [Github link](https://github.com/microsoft/GRAM/tree/main/pretrained_models/CARLA_default)|

## Run Inversion
	sh inversion.sh <CUDA=0> 'z' 'all' 'celebahq' 1000

## Finetune the NeRFGANs
	sh finetune.sh finetune_noise3dlossWMasks celebahq
	
## Rendering results for finetuned models
	sh render_finetuned_imgs.sh 2 'finetune_Z_rec' 'False' 'noise3dloss' 'celebahq'


## Citation
	@inproceedings{yin2023nerfinvertor,
	  title={NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-shot Real Image Animation},
	  author={Yin, Yu and Ghasedi, Kamran and Wu, HsiangTao and Yang, Jiaolong and Tong, Xin and Fu, Yun},
	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	  pages={8539--8548},
	  year={2023}
	}