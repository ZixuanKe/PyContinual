#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o slurm-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_hat_ent_ncl \
    --note random$id \
    --ntasks 10 \
    --nclasses 3 \
    --task dsc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --train_data_size 200 \
    --eval_batch_size 256 \
    --nepochs=100 \
    --lr=0.01 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000
done
#--nepochs 1
#    --train_batch_size 64 \
