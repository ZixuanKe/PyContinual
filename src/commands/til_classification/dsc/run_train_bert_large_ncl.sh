#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_large_dsc_ncl_full_4-%j.out
#SBATCH --gres gpu:1

export TRANSFORMERS_CACHE='/HPS/MultiClassSampling/work/zixuan/model_cache'
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-large-uncased' \
    --note random$id,large\
    --ntasks 10 \
    --task dsc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --train_data_size 200 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --bert_hidden_size 1024
done
#--nepochs 1
#    --train_data_size 500
#    --nclasses 3 \
