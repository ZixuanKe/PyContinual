#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_bert_large_kim_ewc_4-%j.out
#SBATCH --gres gpu:1

export TRANSFORMERS_CACHE='/HPS/MultiClassSampling/work/zixuan/model_cache'
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 4
do
    python  run.py \
    --bert_model 'bert-large-uncased' \
    --experiment bert \
    --approach bert_kim_ewc_ncl \
    --note random$id,large,200,lr0.01 \
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --scenario til_classification \
    --train_batch_size 200 \
    --eval_batch_size 200 \
    --train_data_size 200 \
    --bert_hidden_size 1024
done
#--nepochs 1
#    --nepochs=100 \
#    --lr=0.05 \
#    --lr_min=1e-4 \
#    --lr_factor=3 \
#    --lr_patience=3 \
#    --clipgrad=10000 \
#    --lamb=5000