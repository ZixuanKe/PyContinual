#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
#for id in 0
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment w2v \
    --note random$id \
    --ntasks 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach w2v_kim_hat_ncl
done
#--nepochs 1
