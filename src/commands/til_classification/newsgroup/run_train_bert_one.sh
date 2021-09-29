#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_one_full_4-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_one \
    --eval_batch_size 128 \
    --num_train_epochs 10
done
#--nepochs 1
#    --train_batch_size 64 \
#    --train_data_size 500 \
