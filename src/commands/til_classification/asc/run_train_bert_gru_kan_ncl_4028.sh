#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id \
    --ntasks 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_gru_kan_ncl
done
#--nepochs 1
