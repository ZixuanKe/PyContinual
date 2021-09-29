#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_dsc_adapter_sup_ncl_200-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,sup,t1,200\
    --idrandom $id \
    --ntasks 10 \
    --nclasses 2 \
    --task dsc \
    --scenario dil_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --num_train_epochs 20 \
    --temp 1 \
    --base_temp 1 \
    --sup_loss \
    --train_data_size 200
done
#--nepochs 1
