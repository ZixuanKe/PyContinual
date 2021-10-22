#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1

for id in 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --note random$id,20tasks\
    --ntasks 20 \
    --nclasses 10 \
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_one \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --image_size 32 \
    --image_channel 3 \
    --nepochs 100
done
#--nepochs 1
#    --train_data_size 500
#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,

#    --lr 0.05 \
#    --lr_min 1e-4 \
#    --lr_factor 3 \
#    --lr_patience 5 \
#    --clipgrad 10000