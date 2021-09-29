#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --experiment w2v \
    --note random$id \
    --ntasks 19 \
    --idrandom $id \
    --approach w2v_kim_ncl \
    --eval_batch_size 32
done

    #--nepochs 1
