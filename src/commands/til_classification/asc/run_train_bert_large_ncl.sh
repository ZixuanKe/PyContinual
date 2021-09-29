#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_large_asc_ncl_full_4-%j.out
#SBATCH --gres gpu:1

export TRANSFORMERS_CACHE='/HPS/MultiClassSampling/work/zixuan/model_cache'
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 4
do
     python  run.py \
    --bert_model 'bert-large-uncased' \
    --note random$id,large\
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --learning_rate 1e-5 \
    --bert_hidden_size 1024
done
#--nepochs 1
#    --train_data_size 500
#    --nclasses 3 \
#    --model_path "./models/fp32/til_classification/asc/one_large_$id" \
#    --save_model
#    --model_path "./models/fp32/til_classification/asc/one_large_$id" \
#    --save_model