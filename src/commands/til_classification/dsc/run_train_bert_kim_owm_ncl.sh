#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_bert_kim_owm_full-4%j.out
#SBATCH --gres gpu:1

for id in 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id,full \
    --ntasks 10 \
    --tasknum 10 \
    --task dsc \
    --scenario til_classification \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_kim_owm_ncl \
    --optimizer SGD \
    --clipgrad 100 \
    --lr_min 2e-6 \
    --lr_factor 3 \
    --lr_patience 5 \
    --lr 0.05
done
#--nepochs 1    --train_data_size 200 \
