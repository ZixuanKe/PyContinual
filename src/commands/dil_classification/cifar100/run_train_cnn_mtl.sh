#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_asc_bert_mtl_0-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --note random$id,200,full\
    --ntasks 10 \
    --nclasses 2 \
    --task celeba \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_mtl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --train_data_size 200 \
    --data_size full \
    --image_size 32 \
    --image_channel 3 \
    --mtl
done
#--nepochs 1
