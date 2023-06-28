# NeRFInvertor
NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-shot Real Image Animation, 
[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yin_NeRFInvertor_High_Fidelity_NeRF-GAN_Inversion_for_Single-Shot_Real_Image_Animation_CVPR_2023_paper.pdf)
[ArXiv](https://arxiv.org/abs/2211.17235)
[Project Page](https://yuyin1.github.io/NeRFInvertor_Homepage/)

## Run Inversion
	sh inversion.sh 0 'z' 'all' 'celebahq' 1000

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