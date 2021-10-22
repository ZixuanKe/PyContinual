#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,100,unseen \
    --ntasks 10 \
    --nclasses 2 \
    --task celeba \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach mlp_owm_ncl \
    --train_batch_size 128 \
    --train_data_size 100 \
    --data_size full \
    --image_size 32 \
    --image_channel 3 \
    --nepochs 1000\
    --unseen \
    --ntasks_unseen 10 \
    --ent_id \
    --model_path "./models/fp32/til_classification/celeba/mlp_owm_ncl_$id" \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/celeba/mlp_owm_ncl_$id" \
    --resume_from_task 9 \
    --eval_batch_size 1
done
#--nepochs 1
#lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,