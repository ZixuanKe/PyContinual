#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_adapter_mask_full_1-%j.out
#SBATCH --gres gpu:1

export TRANSFORMERS_CACHE='/HPS/MultiClassSampling/work/zixuan/model_cache'
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 1
do
    python  run.py \
    --bert_model 'bert-large-uncased' \
    --note random$id,large\
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --bert_hidden_size 1024 \
    --model_path "./models/fp32/til_classification/asc/mask_large_$id" \
    --save_model
done

#    --train_data_size 500
#    --model_path "./models/fp32/til_classification/dsc/adapter_mask_full_$id" \
#    --learning_rate 1e-5 \
