#!/bin/bash
for model in 'unet5' 'unet5_residual' 'unet5_contextdilated' 'unet5_scan' 'unet5_recursive' 'unet5_attention' 'unet5_repeat' 'unet5_dense' 'unet5_ddb15x1' 'unet5_ddb15x1_scu' 'unet5_scu';
do
    for epoch in {5..200..5};
    do
        eval "mkdir ./exp/Rain800_${model}_lsgan_1/evaluate_log/"
        eval "python src/evaluate.py --exp_dir exp/Rain800_${model}_lsgan_1/evaluate_log/epoch_${epoch} --data_dir ./data/Rain800/ --generator ${model} --model_file exp/Rain800_${model}_lsgan_1/checkpoints/G_${epoch}.pth --img_splice yx --crop_size 256";
    done
done
