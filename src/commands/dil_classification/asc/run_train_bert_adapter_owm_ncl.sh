#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id \
    --ntasks 19 \
    --nclasses 3 \
    --scenario dil_classification \
    --task asc \
    --tasknum 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_owm_ncl \
    --experiment bert \
    --eval_batch_size 32 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_owm  \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --bert_adapter_size 50
done
#--nepochs 1
