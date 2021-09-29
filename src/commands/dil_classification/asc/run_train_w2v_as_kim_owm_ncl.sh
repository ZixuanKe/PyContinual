#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment w2v_as \
    --note random$id \
    --ntasks 19 \
    --tasknum 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach w2v_kim_owm_ncl \
    --clipgrad 100 \
    --lr_min 2e-6 \
    --lr_factor 3 \
    --lr_patience 5 \
    --lr 0.05
done
#--nepochs 1
#lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,