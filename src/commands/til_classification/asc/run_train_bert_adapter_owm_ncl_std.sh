#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_asc_bert_adapter_owm_4%j.out
#SBATCH --gres gpu:1


for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random0,seed$id \
    --ntasks 19 \
    --task asc \
    --tasknum 19 \
    --idrandom 0 \
    --output_dir './OutputBert' \
    --approach bert_adapter_owm_ncl \
    --scenario til_classification \
    --experiment bert \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_owm  \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --bert_adapter_size 50 \
    --seed $id
done
#--nepochs 1
