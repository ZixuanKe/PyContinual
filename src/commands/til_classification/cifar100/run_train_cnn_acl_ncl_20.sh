#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_cifar10_acl_-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,20tasks \
    --ntasks 20 \
    --task cifar100 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_acl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --image_size 32 \
    --image_channel 3 \
    --nepochs=1000 \
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --alpha 0.5 \
    --beta 0.5\
    --model_path "./models/fp32/til_classification/cifar100/cnn_acl_20_$id" \
    --save_model
done
