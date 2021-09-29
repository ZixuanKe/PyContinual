#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_bert_kim_ncl-4%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_l2_ncl \
    --note random$id,full \
    --ntasks 10 \
    --task dsc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 200 \
    --train_batch_size 200 \
    --lamb 0.5 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000
done
#--nepochs 1
#    --train_data_size 200 \
