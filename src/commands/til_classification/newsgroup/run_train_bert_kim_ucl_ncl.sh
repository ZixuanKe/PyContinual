#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_newsgroup_bert_kim_ucl_0-%j.out
#SBATCH --gres gpu:1

#export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --tasknum 10 \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_kim_ucl_ncl \
    --nepochs 50 \
    --lr=0.01 \
    --ratio 0.5 \
    --beta 0.03 \
    --lr_rho 0.001 \
    --alpha 0.01  \
    --optimizer SGD \
    --clipgrad 100 \
    --lr_min 2e-6 \
    --lr_factor 3 \
    --lr_patience 5
done
#--nepochs 1
# lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,
#--beta 0.03 --ratio 0.5 --lr_rho 0.001 --alpha 0.01