#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 1
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,epoch24,no_tsv_mask\
    --ntasks 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert_adapter_capsule_mask \
    --eval_batch_size 32 \
    --num_train_epochs 10 \
    --apply_bert_output \
    --apply_two_layer_shared \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --no_tsv_mask
done


