#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --experiment w2v_as \
    --note random$id \
    --idrandom $id \
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --approach w2v_kim_derpp_ncl \
    --eval_batch_size 128 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000 \
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --alpha 0.5 \
    --beta 0.5
done

    #--nepochs 1
