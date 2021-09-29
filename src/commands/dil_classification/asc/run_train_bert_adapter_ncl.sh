#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --idrandom $id \
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3
done
#--nepochs 1
