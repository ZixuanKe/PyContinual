#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id \
    --ntasks 10 \
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach cnn_owm_ncl \
    --image_size 32 \
    --image_channel 3 \
    --nepochs=1000\
    --ent_id \
    --model_path "./models/fp32/til_classification/cifar10/cnn_owm_$id" \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/cifar10/cnn_owm_$id" \
    --resume_from_task 9 \
    --eval_batch_size 1
done
#--nepochs 1
#lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,