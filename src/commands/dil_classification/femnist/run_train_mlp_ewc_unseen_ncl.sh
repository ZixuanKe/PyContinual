#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --note random$id,6200,unseen \
    --task femnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_ewc_ncl \
    --lamb=500\
    --unseen \
    --model_path "./models/fp32/dil_classification/femnist/mlp_ewc_$id" \
    --resume_from_file "./models/fp32/dil_classification/femnist/mlp_ewc_$id" \
    --eval_only \
    --eval_batch_size 128
done
#--nepochs 1


#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,lamb=5000
