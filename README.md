# NeRFInvertor
NeRFInvertor: High Fidelity NeRF-GAN Inversion for Single-shot Real Image Animation, https://arxiv.org/abs/2211.17235


## Run Inversion
	sh inversion.sh 0 'z' 'all' 'celebahq' 1000

## Finetune
	sh finetune.sh finetune_noise3dlossWMasks celebahq
	

## Rendering results for finetuned models
	sh render_finetuned_imgs.sh 2 'finetune_Z_rec' 'False' 'noise3dloss' 'celebahq'
