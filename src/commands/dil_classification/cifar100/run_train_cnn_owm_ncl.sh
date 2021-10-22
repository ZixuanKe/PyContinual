#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id \
    --ntasks 10 \
    --nclasses 100 \
    --task cifar100 \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach cnn_owm_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --image_size 32 \
    --image_channel 3 \
    --nepochs=1000
done
#--nepochs 1
#lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,