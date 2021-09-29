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
    --tasknum 10 \
    --task dsc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_kim_ucl_ncl \
    --train_data_size 200 \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --nepochs 50 \
    --ratio 0.125 \
    --beta 0.002 \
    --lr_rho 0.01 \
    --alpha 5 \
    --optimizer SGD
done
#--nepochs 1