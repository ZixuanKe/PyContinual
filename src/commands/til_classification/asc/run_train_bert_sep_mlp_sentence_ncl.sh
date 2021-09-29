#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=2 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert_sep \
    --note random$id \
    --ntasks 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_sep_mlp_sentence_ncl
done
#--nepochs 1
