#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_asc_adapter_mask_trans_sup_128b_0-%j.out
#SBATCH --gres gpu:1

# when you increase the batch_size, batch size still need to increase, so that the number of updated is similar
for id in 0
do
    python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,128b\
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --model_path "./models/fp32/dil_classification/asc/adapter_mask_trans_sup_128b_$id" \
    --approach bert_adapter_mask_trans_sup_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --num_train_epochs 40 \
    --xusemeval_num_train_epochs 40 \
    --bingdomains_num_train_epochs 120 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --temp 1 \
    --base_temp 1
done
#--nepochs 1
#    --train_batch_size 128 \
#    --save_model
