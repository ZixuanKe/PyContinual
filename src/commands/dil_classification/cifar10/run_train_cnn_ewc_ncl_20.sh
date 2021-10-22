#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --note random$id,20tasks \
    --ntasks 20 \
    --nclasses 10 \
    --task cifar10 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_ewc_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --image_size 32 \
    --image_channel 3 \
    --nepochs=1000 \
    --lamb=50
done
#--nepochs 1


#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,lamb=5000
