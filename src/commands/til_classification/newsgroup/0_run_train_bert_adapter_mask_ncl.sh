#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_newsgroup_adapter_mask_0-%j.out
#SBATCH --gres gpu:1

export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 0
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 10
done

#    --train_data_size 500
