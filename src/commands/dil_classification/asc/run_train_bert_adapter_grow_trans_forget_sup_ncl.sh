#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_asc_grow_trans_forget_sup_ncl-4-%j.out
#SBATCH --gres gpu:1

for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --idrandom $id \
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_grow_trans_forget_sup_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_grow \
    --adapter_size 100 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --temp 1 \
    --base_temp 1
done
#--nepochs 1
#    --adapter_size 2000 \
