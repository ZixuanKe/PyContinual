#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_asc_bert_aux_trans_forget_sup_4-%j.out
#SBATCH --gres gpu:1


for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_aux_trans_forget_sup_ncl \
    --model_path "./models/fp32/dil_classification/asc/aux_trans_forget_sup_$id" \
    --aux_model_path "./models/fp32/dil_classification/asc/aux_trans_forget_sup_aux_$id" \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3\
    --temp 1 \
    --base_temp 1 \
    --aux_net \
    --save_model
done
#--nepochs 1
