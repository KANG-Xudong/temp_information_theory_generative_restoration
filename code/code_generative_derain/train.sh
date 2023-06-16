#!/bin/bash
for model in 'unet5' 'unet5_residual' 'unet5_contextdilated' 'unet5_scan' 'unet5_recursive' 'unet5_attention' 'unet5_repeat' 'unet5_dense' 'unet5_ddb15x1' 'unet5_ddb15x1_scu' 'unet5_scu';
do
eval "python src/train.py --exp_dir ./exp/Rain800_${model}_lsgan --data_dir ./data/Rain800/ --generator ${model} --discriminator discriminator --lsgan --img_splice yx --crop_size 256 --random_flip --num_epochs 200 --checkpoint 5 > train_outputs.log";
eval "mv train_outputs.log ./exp/Rain800_${model}_lsgan_1/train_outputs.log";
done
