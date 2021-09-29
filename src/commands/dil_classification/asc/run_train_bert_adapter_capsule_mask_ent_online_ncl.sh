#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 2:00:00
#SBATCH -o dsc_asc_online_-%j.out
#SBATCH --gres gpu:1

for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,online\
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --idrandom $id \
    --scenario dil_classification \
    --output_dir './OutputBert' \
    --model_path "./models/fp32/dil_classification/asc/capsule_mask_ent_$id" \
    --approach bert_adapter_capsule_mask_ent_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_one_layer_shared \
    --eval_batch_size 1 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/dil_classification/asc/capsule_mask_ent_$id" \
    --resume_from_task 18
done

#TODO: check other number of capsules
#    --apply_one_layer_shared for 1,0,3
#    --apply_two_layer_shared for 2,4
#    --train_batch_size 16 \
