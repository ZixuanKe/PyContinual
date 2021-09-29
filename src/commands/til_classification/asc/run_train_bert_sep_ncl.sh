#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

#for id in 0 1 2 3 4
for id in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=2 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id \
    --ntasks 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_sep_sentence_ncl \
    --experiment bert_sep
done
#--nepochs 1
