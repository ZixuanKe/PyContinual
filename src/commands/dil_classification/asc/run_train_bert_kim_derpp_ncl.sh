#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o slurm-%j.out
#SBATCH --gres gpu:1

for id in 0 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_derpp_ncl \
    --note random$id\
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 256 \
    --train_batch_size 64 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000 \
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --alpha 0.5 \
    --beta 0.5
done
