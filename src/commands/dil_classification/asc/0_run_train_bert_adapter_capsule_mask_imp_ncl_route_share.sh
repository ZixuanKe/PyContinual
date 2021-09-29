#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 50:00:00
#SBATCH -o dil_asc_bert_adapter_capsule_mask_route_share_0-%j.out
#SBATCH --gres gpu:1

for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,transfer_route,share,rlarger_share\
    --ntasks 19 \
    --exp 2layer_whole \
    --idrandom $id \
    --nclasses 3 \
    --scenario dil_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ent_ncl \
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
    --larger_as_share
done

#--nepochs 1
#    --train_batch_size 10 \
#    --learning_rate 3e-4

#    --apply_one_layer_shared for 0 is bad
