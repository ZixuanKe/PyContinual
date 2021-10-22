#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_cifar10_acl_-%j.out
#SBATCH --gres gpu:1


for id in  4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,last_id,20tasks \
    --ntasks 20 \
    --nclasses 10 \
    --task cifar10 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_acl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --last_id \
    --model_path "./models/fp32/dil_classification/cifar10/cnn_acl_20_$id" \
    --save_model
done
