#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 2
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,transfer_route,share\
    --ntasks 19 \
    --exp 2layer_whole \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --use_imp \
    --task asc \
    --eval_batch_size 64 \
    --train_batch_size 10 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --transfer_route \
    --share_conv
done

#--nepochs 1
#    --train_batch_size 10 \
#    --learning_rate 3e-4

#    --apply_one_layer_shared for 0 is bad
