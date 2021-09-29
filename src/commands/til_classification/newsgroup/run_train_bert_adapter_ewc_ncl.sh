#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 30:00:00
#SBATCH -o til_newsgroup_bert_adapter_ewc_0-%j.out
#SBATCH --gres gpu:1

export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --output_dir './OutputBert' \
    --note random$id\
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --scenario til_classification \
    --idrandom $id \
    --approach bert_adapter_ewc_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --apply_bert_attention_output \
    --apply_bert_output \
    --build_adapter \
    --lamb 5000 \
    --num_train_epochs 10
done
#--nepochs 1
#    --learning_rate 3e-4 \
