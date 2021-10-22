#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 32:00:00
#SBATCH -o dil_asc_bert_adapter_capsule_mask_0-%j.out
#SBATCH --gres gpu:1

for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 19 \
    --task asc \
    --idrandom $id \
    --nclasses 3 \
    --scenario dil_classification \
    --output_dir './OutputBert' \
    --model_path "./models/fp32/dil_classification/asc/capsule_mask_ent_$id" \
    --approach bert_adapter_capsule_mask_ent_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_one_layer_shared \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --save_model
done

#TODO: check other number of capsules
#    --apply_one_layer_shared for 1,0,3
#    --apply_two_layer_shared for 2,4
#    --train_batch_size 16 \
