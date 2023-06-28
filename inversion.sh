###############################
##        iNVERSION          ##
#                cuda exp_type subject data num_iter
## sh inversion.sh 3 'freq_phase' 'all' 'ffhq' 2000
## sh inversion.sh 0 'freq_phase' '04999' 'ffhq' 2000
## sh inversion.sh 1 'z' 'all' 'ffhq' 1000
## sh inversion.sh 1 'z' '04999 04964' 'ffhq' 1000

## sh inversion.sh 0 'freq_phase' 'all' 'ffhq' 2000
## sh inversion.sh 1 'z' 'all' 'ffhq' 1000


## sh inversion.sh 1 'freq_phase' 'all' 'cat' 2000
## sh inversion.sh 0 'z' 'all' 'cat' 1000

## sh inversion.sh 0 'z' 'all' 'carla' 1000
#  ffhq: '04999 04964 04934 04832 04829 04813 04811 04763 04729 04723 04721 04685 04670 04620 04619'
#        '04604 04585 04567 04550 04548 04408 04355 04352 04339 04213 04181 04095 04066 04059 03973'

#sh inversion.sh 0 'freq_phase' '04999 04964 04934 04832 04829 04813 04811 04763 04729 04723 04721 04685 04670 04620 04619 04604 04585 04567 04550 04548 04408 04355 04352 04339 04213 04181 04095 04066 04059 03973' 'ffhq' 1000
#sh inversion.sh 1 'freq_phase' 'all' 'ffhq' 1000
#sh inversion.sh 2 'z' '04999 04964 04934 04832 04829 04813 04811 04763 04729 04723 04721 04685 04670 04620 04619 04604 04585 04567 04550 04548 04408 04355 04352 04339 04213 04181 04095 04066 04059 03973' 'ffhq' 1000
#sh inversion.sh 3 'z' 'all' 'ffhq' 1000

#sh inversion.sh 1 'freq_phase' 'all' 'cat' 1000
#sh inversion.sh 3 'z' 'all' 'cat' 1000

#sh inversion.sh 1 'freq_phase' 'all' 'carla' 2000
#sh inversion.sh 3 'z' 'all' 'carla' 1000

###############################
#max_iter=2000
#max_iter=1000

if ! [ -z "$1" ]
then
  CUDA_VISIBLE_DEVICES=$1
else
  CUDA_VISIBLE_DEVICES=0
fi
if ! [ -z "$2" ]
then
  opt_param=$2
else
  opt_param='freq_phase'
fi

if ! [ -z "$3" ]
then
  target_names=$3
else
  target_names='all'
fi

if ! [ -z "$4" ]
then
  dataset=$4
else
  dataset='ffhq'
fi
if ! [ -z "$5" ]
then
  max_iter=$5
else
  max_iter=2000
fi


###############################
##           FFHQ            ##
###############################
if [ $dataset = "ffhq" ]
then
  output_dir='../exp_gram/inversion_ffhq'
  generator_file='../pretrained_models/gram/FFHQ_default/generator.pth'
  img_dir='../../Dataset/NeRFGAN/image256_align_new_mirror_wo_t/'
  mat_dir='../../Dataset/NeRFGAN/ffhq_pose_align_new_mirror/'
  config='FFHQ_default'
  lambda_perceptual=1
  lambda_l2=0.01
  lambda_id=0.01
  lambda_reg=0.04
elif [ $dataset = "celebahq" ]
then
  output_dir='../exp_gram/inversion_celebahq'
  generator_file='../pretrained_models/gram/FFHQ_default/generator.pth'
  img_dir='../../Dataset/NeRFGAN/celebahq_test256_align_new_mirror_wo_t/'
  mat_dir='../../Dataset/NeRFGAN/celebahq_test256_mat/'
  config='CelebAHQ_default'
  lambda_perceptual=1
  lambda_l2=0.01
  lambda_id=0.01
  lambda_reg=0.04
elif [ $dataset = "toon" ]
then
  output_dir='../exp_gram/inversion_toon'
  generator_file='../pretrained_models/gram/FFHQ_default/generator.pth'
  img_dir='../../Dataset/NeRFGAN/toon_pairs_share_align_new_mirror_wo_t1/'
  mat_dir='../../Dataset/NeRFGAN/toon_pairs_share_mat/'
  config='FFHQ_default'
  lambda_perceptual=1
  lambda_l2=0.01
  lambda_id=0.01
  lambda_reg=0.04
elif [ $dataset = "multiEdits" ]
then
  output_dir='../exp_gram/inversion_multiEdits'
  generator_file='../pretrained_models/gram/FFHQ_default/generator.pth'
  img_dir='../../Dataset/NeRFGAN/multiEdits_align_new_mirror_wo_t1/'
  mat_dir='../../Dataset/NeRFGAN/multiEdits_mat/'
  config='FFHQ_default'
  lambda_perceptual=1
  lambda_l2=0.01
  lambda_id=0.01
  lambda_reg=0.04
elif [ $dataset = "cat" ]
then
  output_dir='../exp_gram/inversion_cat'
  generator_file='../pretrained_models/gram/CATS_default/generator.pth'
  img_dir='../../Dataset/NeRFGAN/cats2_256/'
  mat_dir='../../Dataset/NeRFGAN/cats2/poses/'
  config='CATS_default'
  lambda_perceptual=1
  lambda_l2=0.01
  lambda_id=0
  lambda_reg=0.04
elif [ $dataset = "carla" ]
then
  output_dir='../exp_gram/inversion_carla'
  generator_file='../pretrained_models/gram/CARLA_default/generator.pth'
  img_dir='../../Dataset/NeRFGAN/carla128/'
  mat_dir='../../Dataset/NeRFGAN/carla/carla_poses/'
  config='CARLA_default'
  lambda_perceptual=1
  lambda_l2=0.1
  lambda_id=0
  lambda_reg=0.04
fi
if [ $opt_param = "freq_phase" ]
    then
      output_dir=$output_dir'_FreqPhase'
    elif [ $opt_param = "z" ]
    then
      output_dir=$output_dir'_Z'
fi
if [ $target_names = "all" ]
then
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inversion.py \
      --generator_file $generator_file \
      --output_dir $output_dir \
      --img_dir $img_dir \
      --mat_dir $mat_dir \
      --config $config \
      --opt_param $opt_param \
      --sv_interval 20 \
      --lambda_perceptual $lambda_perceptual \
      --lambda_l2 $lambda_l2 \
      --lambda_id $lambda_id\
      --lambda_reg $lambda_reg \
      --max_iter $max_iter
else
  for target_name in $target_names
  do
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inversion.py \
        --generator_file $generator_file \
        --output_dir $output_dir \
        --img_dir $img_dir \
        --mat_dir $mat_dir \
        --config $config \
        --opt_param $opt_param \
        --sv_interval 20 \
        --lambda_perceptual $lambda_perceptual \
        --lambda_l2 $lambda_l2 \
        --lambda_id $lambda_id\
        --lambda_reg $lambda_reg \
        --name $target_name \
        --max_iter $max_iter
  done
fi





