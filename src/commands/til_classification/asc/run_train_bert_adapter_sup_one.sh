#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_asc_adapter_tmix_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,sup\
    --idrandom $id \
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_one \
    --experiment bert \
    --eval_batch_size 128 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --temp 1 \
    --base_temp 1 \
    --output_dir './OutputBert' \
    --sup_loss
done
#--nepochs 1
