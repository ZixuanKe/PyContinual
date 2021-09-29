#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id \
    --ntasks 10 \
    --task dsc \
    --train_data_size 200 \
    --eval_batch_size 64 \
    --train_batch_size 64 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_gru_srk_ncl
done
#--nepochs 1
