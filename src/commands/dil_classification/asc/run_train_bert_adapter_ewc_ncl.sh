#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-base-uncased' \
    --output_dir './OutputBert' \
    --note random$id\
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --idrandom $id \
    --approach bert_adapter_ewc_ncl \
    --experiment bert \
    --eval_batch_size 32 \
    --train_batch_size 32 \
    --apply_bert_attention_output \
    --apply_bert_output \
    --build_adapter \
    --lamb 5000 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3
done
#--nepochs 1
#    --learning_rate 3e-4 \
