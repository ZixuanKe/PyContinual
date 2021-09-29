#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  main.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id,two_thirds \
    --ntasks 19 \
    --tasknum 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_kim_owm_ncl \
    --optimizer SGD \
    --cut_partition 0.666
done
#--nepochs 1