#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_adapter_mask_full_0-%j.out
#SBATCH --gres gpu:1

#export TRANSFORMERS_CACHE='/HPS/MultiClassSampling/work/zixuan/model_cache'
#export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

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
    --approach bert_adapter_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --train_data_size 200 \
    --num_train_epochs 20 \
    --bert_hidden_size 1024
done

#    --train_data_size 500
#    --model_path "./models/fp32/til_classification/dsc/adapter_mask_full_$id" \
#    --learning_rate 1e-5 \
