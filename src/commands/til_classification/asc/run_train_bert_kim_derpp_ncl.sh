#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in 4
do
     python run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_derpp_ncl \
    --note random$id\
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 256 \
    --train_batch_size 128 \
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
