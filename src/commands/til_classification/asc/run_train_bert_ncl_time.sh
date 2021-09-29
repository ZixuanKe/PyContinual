#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_bert_ncl-4%j.out
#SBATCH --gres gpu:1

for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --task asc \
    --scenario til_classification \
    --ntasks 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_ncl \
    --experiment bert \
    --eval_batch_size 32 \
    --num_train_epochs 1 \
    --xusemeval_num_train_epochs 1 \
    --bingdomains_num_train_epochs 1 \
    --bingdomains_num_train_epochs_multiplier 1 \
    --exit_after_first_task
done
#--nepochs 1
