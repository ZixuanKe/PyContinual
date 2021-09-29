#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python run.py \
    --experiment w2v_as \
    --note random0,sup,seed$id \
    --idrandom 0 \
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario til_classification \
    --approach w2v_kim_one \
    --eval_batch_size 128 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000 \
    --seed $id
done

#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000
