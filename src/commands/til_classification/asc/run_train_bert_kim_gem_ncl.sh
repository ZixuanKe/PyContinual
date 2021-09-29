#!/bin/bash



#TODO: GEM baseline, could be time-consuming, consider A-GEM
for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_gem_ncl \
    --note random$id \
    --ntasks 19 \
    --task asc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --gamma 0.5 \
    --train_batch_size 200 \
    --eval_batch_size 200
done
#--nepochs 1
