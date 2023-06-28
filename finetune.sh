# sh finetune.sh finetune_noise3dloss celebahq

# CUDA_VISIBLE_DEVICES='0,1'
CUDA_VISIBLE_DEVICES='0,1,2,3'
if ! [ -z "$1" ]
then
  exp_type=$1
else
  exp_type='finetune_noise3dlossWMasks'
fi

if ! [ -z "$2" ]
then
  dataset=$2
else
  dataset='celebahq'
fi



if [ $dataset = "ffhq" ]
  then
    subjects="00000+00001"
    config='FFHQ_finetune'
    load_dir='../pretrained_models/'
    output_dir='../exp_gram/trained/ffhq'
    data_img_dir='../../Dataset/NeRFGAN/image256_align_new_mirror_wo_t'
    data_pose_dir='../../Dataset/NeRFGAN/ffhq_pose_align_new_mirror'
    data_emd_dir='../exp_gram/inversion_ffhq'
    pretrain_model='gram/FFHQ_default/'
    lambda_id=0.1
    regulizer_alpha=5
  elif [ $dataset = "celebahq" ]
  then
    subjects='000080+000143'
    config='CelebAHQ_finetune'
    load_dir='../pretrained_models/'
    output_dir='../exp_gram/trained/celebahq'
    data_img_dir='../../Dataset/NeRFGAN/celebahq_test256_align_new_mirror_wo_t'
    data_pose_dir='../../Dataset/NeRFGAN/celebahq_test256_mat'
    data_emd_dir='../exp_gram/inversion_celebahq'
    pretrain_model='gram/FFHQ_default/'
    lambda_id=0.1
    regulizer_alpha=5
  elif [ $dataset = "toon" ]
  then
    subjects='start_from_58'
    # subjects='02089_org+02089_pnt+02100_org+02100_pnt+02206_org'
    config='FFHQ_finetune'
    load_dir='../pretrained_models/'
    output_dir='../exp_gram/trained/toon'
    data_img_dir='../../Dataset/NeRFGAN/toon_pairs_share_align_new_mirror_wo_t1'
    data_pose_dir='../../Dataset/NeRFGAN/toon_pairs_share_mat'
    data_emd_dir='../exp_gram/inversion_toon'
    pretrain_model='gram/FFHQ_default/'
    lambda_id=0.1
    regulizer_alpha=5
  elif [ $dataset = "multiEdits" ]
  then
    subjects='start_from_0'
    # subjects='+'
    config='FFHQ_finetune'
    load_dir='../pretrained_models/'
    output_dir='../exp_gram/trained/multiEdits'
    data_img_dir='../../Dataset/NeRFGAN/multiEdits_align_new_mirror_wo_t1'
    data_pose_dir='../../Dataset/NeRFGAN/multiEdits_mat'
    data_emd_dir='../exp_gram/inversion_multiEdits'
    pretrain_model='gram/FFHQ_default/'
    lambda_id=0.1
    regulizer_alpha=5
  elif [ $dataset = "cat" ]
  then
    subjects='00000001_000+00000001_005'
    config='CATS_finetune'
    load_dir='../pretrained_models/'
    output_dir='../exp_gram/trained/cat'
    data_img_dir='../../Dataset/NeRFGAN/cats2_256'
    data_pose_dir='../../Dataset/NeRFGAN/cats2/poses'
    data_emd_dir='../exp_gram/inversion_cat'
    pretrain_model='gram/CATS_default/'
    lambda_id=0
    regulizer_alpha=10
  elif [ $dataset = "carla" ]
  then
    subjects='000000+000001+000002+000003+000004+000005+000006+000007+000008+000009+000010'
    config='CARLA_finetune'
    load_dir='../pretrained_models/'
    output_dir='../exp_gram/trained/carla'
    data_img_dir='../../Dataset/NeRFGAN/carla128'
    data_pose_dir='../../Dataset/NeRFGAN/carla/carla_poses'
    data_emd_dir='../exp_gram/inversion_carla'
    pretrain_model='gram/CARLA_default/'
    lambda_id=0
    regulizer_alpha=0.1
fi


if [ $exp_type = "finetune_noise3dloss" ]
then
    # with reg, 3dloss, noise inverison
    data_emd_dir=$data_emd_dir'_Z/'
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python finetune.py \
        --target_names $subjects \
        --config $config \
        --load_dir $load_dir \
        --output_dir $output_dir \
        --data_img_dir $data_img_dir \
        --data_pose_dir $data_pose_dir \
        --data_emd_dir $data_emd_dir \
        --pretrain_model $pretrain_model \
        --loc_reg \
        --target_inv_epoch 00999 \
        --regulizer_alpha $regulizer_alpha \
        --lambda_id $lambda_id \
        --lambda_reg_rgbBefAggregation 10 \
        --experiment_name $exp_type
elif [ $exp_type = "finetune_noise3dlossWMasks" ]
then
    # with reg, 3dloss, with mask, noise inverison
    data_emd_dir=$data_emd_dir'_Z/'
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python finetune.py \
        --target_names $subjects \
        --config $config \
        --load_dir $load_dir \
        --output_dir $output_dir \
        --data_img_dir $data_img_dir \
        --data_pose_dir $data_pose_dir \
        --data_emd_dir $data_emd_dir \
        --pretrain_model $pretrain_model \
        --loc_reg \
        --load_mat \
        --target_inv_epoch 00999 \
        --regulizer_alpha $regulizer_alpha \
        --lambda_id $lambda_id \
        --lambda_reg_rgbBefAggregation 10 \
        --lambda_bg_sigma 10 \
        --experiment_name $exp_type
elif [ $exp_type = "finetune_noisepose3dloss" ]
then
    # with reg, 3dloss, noise_pose inverison
    data_emd_dir='../experiments/inversion_ffhq_noise_pose/'
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python finetune.py \
        --target_names $subjects \
        --load_dir $load_dir \
        --output_dir $output_dir \
        --data_img_dir $data_img_dir \
        --data_pose_dir $data_pose_dir \
        --data_emd_dir $data_emd_dir \
        --loc_reg \
        --sample_noise \
        --sample_id \
        --sample_exp \
        --sample_pose \
        --target_inv_epoch 01999 \
        --lambda_reg_rgbBefAggregation 10 \
        --experiment_name $exp_type
fi


