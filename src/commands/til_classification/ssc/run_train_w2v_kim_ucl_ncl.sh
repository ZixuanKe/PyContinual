#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  main.py \
    --bert_model 'bert-base-uncased' \
    --experiment w2v \
    --note random$id \
    --ntasks 10 \
    --task dsc \
    --tasknum 10 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach w2v_kim_ucl_ncl \
    --ratio 0.125 \
    --beta 0.002 \
    --lr_rho 0.01 \
    --alpha 5 \
    --lr 0.05 \
    --optimizer SGD \
    --train_data_size 200 \
    --train_batch_size 128 \
    --eval_batch_size 128
done
#--nepochs 1
# --beta 0.0002 --ratio 0.125 --lr_rho 0.01 --alpha 0.3