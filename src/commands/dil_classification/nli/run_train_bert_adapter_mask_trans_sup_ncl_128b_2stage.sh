#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_nli_adapter_mask_sup_trans_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,sup,trans,128b,2000\
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mask_sup_ncl_2stage \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --num_train_epochs 40 \
    --train_data_size 2000 \
    --dev_data_size 2000 \
    --temp 1 \
    --base_temp 1 \
    --trans_loss \
    --sup_loss \
    --two_stage
done
#--nepochs 1
#    --save_model
#    --sup_loss
