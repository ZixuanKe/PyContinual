#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1

for id in 2
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --note random$id\
    --ntasks 10 \
    --nclasses 10 \
    --task cifar10 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --image_size 32 \
    --image_channel 3 \
    --nepochs 1000
done
#--nepochs 1
#    --train_data_size 500
#lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,
#    --sup_loss

#    --lr 0.05 \
#    --lr_min 1e-4 \
#    --lr_factor 3 \
#    --lr_patience 5 \
#    --clipgrad 10000 \