#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_ncl_full_4-%j.out
#SBATCH --gres gpu:1
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/ASC/dataset_cache'

for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id\
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_mtl \
    --eval_batch_size 128 \
    --num_train_epochs 10
done
#--nepochs 1
#    --train_data_size 500
#    --nclasses 3 \
