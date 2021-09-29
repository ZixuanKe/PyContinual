#!/bin/bash

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,2000\
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_sup_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --train_data_size 2000 \
    --dev_data_size 2000 \
    --temp 1 \
    --base_temp 1
done
#--nepochs 1
