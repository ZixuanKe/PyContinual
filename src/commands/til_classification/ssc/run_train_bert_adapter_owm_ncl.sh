#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id \
    --ntasks 10 \
    --task dsc \
    --tasknum 10 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_owm_ncl \
    --experiment bert \
    --train_data_size 200 \
    --eval_batch_size 32 \
    --train_batch_size 28 \
    --num_train_epochs 20 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_owm  \
    --adapter_size 50
done
#--nepochs 1
