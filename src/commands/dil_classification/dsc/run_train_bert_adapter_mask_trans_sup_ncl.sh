#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 12:00:00
#SBATCH -o dil_dsc_adapter_mask_sup_trans_2000_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,trans,sup,200\
    --ntasks 10 \
    --nclasses 2 \
    --task dsc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mask_sup_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --train_data_size 200 \
    --num_train_epochs 20 \
    --temp 1 \
    --base_temp 1 \
    --trans_loss \
    --sup_loss
done
#--nepochs 1
