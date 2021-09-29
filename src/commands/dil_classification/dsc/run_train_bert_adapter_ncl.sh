#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_dsc_bert_adapter_ncl_2000_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id,2000\
    --ntasks 10 \
    --nclasses 2 \
    --task dsc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_ncl \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --eval_batch_size 128 \
    --train_data_size 2000 \
    --num_train_epochs 20
done
#--nepochs 1
#    --train_data_size 200
#    --model_path "./models/fp32/dil_classification/dsc/adapter_ncl_full_$id" \
#    --save_model
