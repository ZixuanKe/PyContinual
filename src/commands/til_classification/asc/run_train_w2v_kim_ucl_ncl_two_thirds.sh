#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  main.py \
    --bert_model 'bert-base-uncased' \
    --experiment w2v \
    --note random$id,two_thirds \
    --ntasks 19 \
    --tasknum 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach w2v_kim_ucl_ncl \
    --ratio 0.125 \
    --beta 0.002 \
    --lr_rho 0.01 \
    --alpha 5 \
    --lr 0.05 \
    --optimizer SGD \
    --cut_partition 0.666
done
#--nepochs 1
# --beta 0.0002 --ratio 0.125 --lr_rho 0.01 --alpha 0.3