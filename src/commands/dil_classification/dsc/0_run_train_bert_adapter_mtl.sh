#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_dsc_bert_adapter_mtl_full_0-%j.out
#SBATCH --gres gpu:1


for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id,full \
    --ntasks 10 \
    --nclasses 3 \
    --task dsc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mtl \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --eval_batch_size 128 \
    --num_train_epochs 20
done
#--nepochs 1
#    --train_data_size 500 \
#    --save_model
#    --model_path "./models/fp32/dil_classification/dsc/adapter_mtl_full_$id" \
