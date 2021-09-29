#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_capsule_mask_imp_route_share_3-%j.out
#SBATCH --gres gpu:1

for id in 3
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,transfer_route,share,rlarger_share,adapter768\
    --ntasks 19 \
    --exp 2layer_whole \
    --idrandom $id \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --use_imp \
    --task asc \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --transfer_route \
    --share_conv \
    --larger_as_share \
    --adapter_size 768\
    --model_path "./models/fp32/til_classification/asc/capsule_mask_imp_route_share_768_$id" \
    --save_model \
    --resume_model \
    --resume_from_file "./models/fp32/til_classification/asc/capsule_mask_imp_route_share_768_$id" \
    --resume_from_task 9
done
