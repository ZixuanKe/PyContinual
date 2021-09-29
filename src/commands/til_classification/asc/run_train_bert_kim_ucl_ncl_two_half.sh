#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  main.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id,half \
    --ntasks 19 \
    --tasknum 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_kim_ucl_ncl \
    --ratio 0.125 \
    --beta 0.002 \
    --lr_rho 0.01 \
    --alpha 5 \
    --optimizer SGD \
    --cut_partition 0.5
done
#--nepochs 1