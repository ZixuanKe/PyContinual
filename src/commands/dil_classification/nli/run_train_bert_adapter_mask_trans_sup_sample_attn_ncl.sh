#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_nli_adapter_mask_sup_trans_4-%j.out
#SBATCH --gres gpu:1


for id in 0
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,re_forward,sample_att,sup,trans,2000\
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mask_sup_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --train_data_size 2000 \
    --dev_data_size 2000 \
    --temp 1 \
    --base_temp 1 \
    --trans_loss \
    --sup_loss \
    --output_sample_gate \
    --re_forward
done
#--nepochs 1
#    --model_path "./models/fp32/dil_classification/nli/adapter_mask_trans_sup_2000_$id" \
#    --save_model
#    --sup_loss
#    --output_pooled_rep \
