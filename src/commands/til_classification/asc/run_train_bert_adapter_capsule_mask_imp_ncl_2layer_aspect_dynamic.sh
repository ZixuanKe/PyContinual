#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,2layer_aspect_dynamic\
    --ntasks 19 \
    --task asc \
    --exp 2layer_aspect_dynamic \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert \
    --eval_batch_size 32 \
    --train_batch_size 20 \
    --num_train_epochs 10 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --use_imp \
    --max_seq_length 133
done

#--nepochs 1
#    --apply_one_layer_shared for 1,0,3
#    --apply_two_layer_shared for 2,4
