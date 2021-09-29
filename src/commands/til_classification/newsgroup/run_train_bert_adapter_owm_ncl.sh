#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_newsgroup_bert_adapter_kim_owm_0-%j.out
#SBATCH --gres gpu:1

export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --scenario til_classification \
    --tasknum 10 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_owm_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_owm  \
    --bert_adapter_size 50
done
#--nepochs 1
