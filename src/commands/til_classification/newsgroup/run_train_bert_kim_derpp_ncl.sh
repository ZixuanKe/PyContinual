#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o til_newsgroup_kim_derpp_4-%j.out
#SBATCH --gres gpu:1
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 0 1 2 3 4
do
     python run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_derpp_ncl \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000 \
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --alpha 0.5 \
    --beta 0.5
done
#    --train_data_size 200 \
