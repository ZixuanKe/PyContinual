#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 30:00:00
#SBATCH -o til_newsgroup_bert_kim_hat_0-%j.out
#SBATCH --gres gpu:1

#export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_kim_hat_ncl \
    --nepochs=100 \
    --lr=0.2 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000
done
#--nepochs 1
