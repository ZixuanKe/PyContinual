#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id \
    --ntasks 20 \
    --nclasses 10 \
    --task cifar10 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_ucl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --image_size 32 \
    --image_channel 3 \
    --nepochs 1000 \
    --ratio 0.125 \
    --beta 0.002 \
    --lr_rho 0.01 \
    --alpha 5 \
    --optimizer SGD \
    --clipgrad 100 \
    --lr_min 2e-6 \
    --lr_factor 3 \
    --lr_patience 5
done
#--nepochs 1
# lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,