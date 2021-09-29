#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_newsgroup_bert_gru_srk_4-%j.out
#SBATCH --gres gpu:1

export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id \
    --ntasks 10 \
    --scenario til_classification \
    --class_per_task 2 \
    --task newsgroup \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_gru_srk_ncl \
    --train_batch_size 200 \
    --eval_batch_size 200
done
#--nepochs 1
#    --train_data_size 200 \
