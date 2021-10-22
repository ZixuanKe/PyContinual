#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,6200 \
    --task femnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_ucl_ncl \
    --unseen \
    --model_path "./models/fp32/dil_classification/femnist/mlp_ucl_$id" \
    --resume_from_file "./models/fp32/dil_classification/femnist/mlp_ucl_$id" \
    --eval_only \
    --eval_batch_size 128
done
#--nepochs 1
# lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,