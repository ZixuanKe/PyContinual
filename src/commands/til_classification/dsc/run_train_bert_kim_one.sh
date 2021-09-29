#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_dsc_bert_kim_one-4%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_one \
    --note random$id,full \
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --nepochs=100 \
    --lr=0.01 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000
done
#--nepochs 1
    --train_data_size 200 \
