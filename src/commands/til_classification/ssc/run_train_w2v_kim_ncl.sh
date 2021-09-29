#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --experiment w2v \
    --note random$id \
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --approach w2v_kim_ncl \
    --train_data_size 200 \
    --eval_batch_size 128 \
    --train_batch_size 128
done

    #--nepochs 1
