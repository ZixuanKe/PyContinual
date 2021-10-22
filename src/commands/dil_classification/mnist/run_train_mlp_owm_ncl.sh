#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 2
do
    CUDA_VISIBLE_DEVICES=1 python run.py \
    --note random$id \
    --ntasks 10 \
    --nclasses 10 \
    --task mnist \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach mlp_owm_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --image_size 28 \
    --image_channel 1 \
    --nepochs 1000
done
#--nepochs 1
#lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,