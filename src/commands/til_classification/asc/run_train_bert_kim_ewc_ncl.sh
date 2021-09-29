#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id \
    --ntasks 19 \
    --task asc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_kim_ewc_ncl
done
#--nepochs 1
#    --nepochs=100 \
#    --lr=0.05 \
#    --lr_min=1e-4 \
#    --lr_factor=3 \
#    --lr_patience=3 \
#    --clipgrad=10000 \
#    --lamb=5000