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
    subjects="00000+00001+00002+00003+00004+00005+00006+00007+00008+00009+00010+00011+00012+00013+00014+00015+00016+00017+00018+00019+"
    subjects=$subjects'00020+00021+00022+00023+00024+00025+00026+00027+00028+00029+00030+00031+00032+00033+00034+00035+00036+00037+00038+00039+'
    subjects=$subjects'00040+00041+00042+00043+00044+00045+00046+00047+00048+00049+00050+00051+00052+00053+00054+00055+00056+00057+00058+00059+'
    subjects=$subjects'00060+00061+00062+00063+00064+00065+00066+00067+00068+00069+00070+00071+00072+00073+00074+00075+00076+00077+00078+00079+'
    subjects=$subjects'00080+00081+00082+00083+00084+00085+00086+00087+00088+00089+00090+00091+00092+00093+00094+00095+00096+00097+00098+00099+'

    # subjects='00082+00083+00084+00085+00086+00087+00088+00089+00090+00091+00092+00093+00094+00095+00096+00097+00098+00099+'
    # subjects=$subjects'00100+00101+00102+00103+00104+00105+00106+00107+00108+00109+00110+00111+00112+00113+00114+00115+00116+00117+00118+00119+'
    # subjects=$subjects'00120+00121+00122+00123+00124+00125+00126+00127+00128+00129+00130+00131+00132+00133+00134+00135+00136+00137+00138+00139+'
    # subjects=$subjects'00140+00141+00142+00143+00144+00145+00146+00147+00148+00149'
    # subjects='00151+00399+00485+00494+00692+00771+00855+00881+00911+00912+00960+01020+01974+02330+02402+02483+02487+02582+02828+03288'
    # subjects=$subjects'+04999+04964+04934+04829+04811+04763+04723+04721+04685+04670+04620+04619+04604+04585+04567+04550+04548+04408+04355+04352'
    # subjects=$subjects'+04339+04213+04181+04095+04066+04059+03973'
    # 00000+00001+00002+00003+00004+00005+00006+00007+00008+00009+00010+00011+00012+00013+00014+00015+00016+00017+00018+00019+
    # 00020+00021+00022+00023+00024+00025+00026+00027+00028+00029+00030+00031+00032+00033+00034+00035+00036+00037+00038+00039+
    # 00040+00041+00042+00043+00044+00045+00046+00047+00048+00049+00050+00051+00052+00053+00054+00055+00056+00057+00058+00059+
    # 00060+00061+00062+00063+00064+00065+00066+00067+00068+00069+00070+00071+00072+00073+00074+00075+00076+00077+00078+00079+
    # 00080+00081+00082+00083+00084+00085+00086+00087+00088+00089+00090+00091+00092+00093+00094+00095+00096+00097+00098+00099+
    # 00100+00101+00102+00103+00104+00105+00106+00107+00108+00109+00110+00111+00112+00113+00114+00115+00116+00117+00118+00119+
    # 00120+00121+00122+00123+00124+00125+00126+00127+00128+00129+00130+00131+00132+00133+00134+00135+00136+00137+00138+00139+
    # 00140+00141+00142+00143+00144+00145+00146+00147+00148+00149
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
    # subjects='000080+000143+000168+000196+000406+000431+000442+000443+000446+000528+000534+000622+000656+000801+000990+'
    # subjects=$subjects'001022+001036+001074+001092+001349+001496+001663+001693+001743+001804+001972+002064+002110+002123+002140+'
    # subjects=$subjects'002292+002317+002378+002419+002420+002475+002528+002569+002708+002768+002963+003247+003251+003309+003536+'
    # subjects=$subjects'003569+003622+003675+003716+003767+003789+003959+004056+004070+004253+004480+004487+004526+004603+004666+'
    # subjects=$subjects'004715+004730+004763+004858+004892+004911+005061+005134+005235+005387+005503+005539+005586+005664+005670+'
    # subjects=$subjects'005703+005735+005799+005846+005923+005926+006011+006012+006085+006087+006105+006595+006636+006720+006750+'
    # subjects=$subjects'006877+006878+006930+007033+007072+007075+007143+007150+007156+007224+007250+007398+007425+007536+007710+'
    # subjects=$subjects'007798+007810+007817+007821+008113+008149+008210+008322+008472+008548+008550+008582+008604+008731+008843+'
    # subjects=$subjects'008966+008994+009120+009250+009480+009556+009644+009661+009698+009737+009742+009858+009882+010011+010092+'
    # subjects=$subjects'010151+010168+010169+010203+010239+010264+010309+010441+010492+010551+010588+010791+010839+010866+010961'
    # 000080+000143+000168+000196+000406+000431+000442+000443+000446+000528+000534+000622+000656+000801+000990+
    # 001022+001036+001074+001092+001349+001496+001663+001693+001743+001804+001972+002064+002110+002123+002140+
    # 002292+002317+002378+002419+002420+002475+002528+002569+002708+002768+002963+003247+003251+003309+003536+
    # 003569+003622+003675+003716+003767+003789+003959+004056+004070+004253+004480+004487+004526+004603+004666+
    # 004715+004730+004763+004858+004892+004911+005061+005134+005235+005387+005503+005539+005586+005664+005670+
    # 005703+005735+005799+005846+005923+005926+006011+006012+006085+006087+006105+006595+006636+006720+006750+
    # 006877+006878+006930+007033+007072+007075+007143+007150+007156+007224+007250+007398+007425+007536+007710+
    # 007798+007810+007817+007821+008113+008149+008210+008322+008472+008548+008550+008582+008604+008731+008843+
    # 008966+008994+009120+009250+009480+009556+009644+009661+009698+009737+009742+009858+009882+010011+010092+
    # 010151+010168+010169+010203+010239+010264+010309+010441+010492+010551+010588+010791+010839+010866+010961

    subjects='011629+011696+013340+013461+013881+014015+014977+015088+015940+016389+019974+020298+023620+024869+026591+031261+039149+045092+059839+060259+081440+097665+161788'
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
    subjects='00000001_000+00000001_005+00000001_017+00000001_020+00000001_024+00000002_001+00000002_003+00000004_008+00000004_012+00000005_001'
    subjects=$subjects'+00000005_017+00000005_020+00000005_022+00000006_009+00000006_011+00000006_022+00000006_024+00000007_000+00000007_002+00000008_007'
    subjects=$subjects'+00000009_003+00000009_004+00000009_005+00000009_013+00000011_020+00000013_011+00000013_021+00000022_028+00000023_020+00000024_024'
    subjects=$subjects'+00000025_015+00000028_024+00000031_019+00000069_009+00000118_022+00000122_021+00000133_000+00000134_010+00000134_021+00000142_006'
    # subjects='00000069_009+00000118_022+00000122_021+00000133_000+00000134_010+00000134_021+00000142_006'
    subjects=$subjects'+00000142_027+00000150_013+00000174_027+00000193_005+00000193_008+00000193_024+00000208_003+00000210_012+00000212_006+00000214_003'
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
    subjects=$subjects'+000011+000012+000013+000014+000015+000016+000017+000018+000019+000020'
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


if [ $exp_type = "finetune_woreg" ]
then
    # without reg loss
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
        --target_inv_epoch 00999 \
        --lambda_id $lambda_id \
        --lambda_loc_reg_l2 0 \
        --lambda_loc_reg_perceptual 0 \
        --lambda_reg_volumeDensity 0 \
        --lambda_reg_rgbBefAggregation 0 \
        --lambda_reg_sigmaBefAggregation 0 \
        --experiment_name $exp_type
elif [ $exp_type = "finetune_noise2dloss" ]
then
    # with reg, 2dloss, noise inverison
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
        --lambda_reg_rgbBefAggregation 0 \
        --experiment_name $exp_type
elif [ $exp_type = "finetune_noise3dloss" ]
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


