#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_nli_adapter_mask_sup_trans_4-%j.out
#SBATCH --gres gpu:1


for id in 0
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,forget,sup,1000\
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_grow_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_grow \
    --eval_batch_size 128 \
    --num_train_epochs 20 \
    --train_data_size 1000 \
    --dev_data_size 1000 \
    --temp 1 \
    --base_temp 1 \
    --distill_loss \
    --sup_loss \
    --s_dim 3 \
    --t_dim 3
done
#--nepochs 1
#    --model_path "./models/fp32/dil_classification/nli/adapter_mask_trans_sup_2000_$id" \
#    --save_model
#    --sup_loss
