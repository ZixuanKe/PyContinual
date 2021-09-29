#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 2 3 4
do
    CUDA_VISIBLE_DEVICES=2 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,epoch24\
    --ntasks 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert_adapter_capsule_mask \
    --eval_batch_size 32 \
    --num_train_epochs 10 \
    --apply_bert_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --train_batch_size 24

done