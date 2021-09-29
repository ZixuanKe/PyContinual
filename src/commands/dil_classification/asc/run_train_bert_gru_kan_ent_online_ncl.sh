#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id,online,ent_id \
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_gru_kan_ncl \
    --eval_batch_size 1 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000 \
    --ent_id \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/dil_classification/asc/bert_gru_kan_ent_$id" \
    --resume_from_task 18 \
    --model_path "./models/fp32/dil_classification/asc/bert_gru_kan_ent_$id"
done
#--nepochs 1
#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,