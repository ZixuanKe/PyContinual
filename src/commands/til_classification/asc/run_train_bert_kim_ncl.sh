#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_ncl \
    --note random$id \
    --ntasks 19 \
    --task asc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 256 \
    --train_batch_size 256 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000
done
#--nepochs 1
