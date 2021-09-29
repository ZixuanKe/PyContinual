#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_nli_adapter_mask_forget_sup_4-%j.out
#SBATCH --gres gpu:1


for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,2000\
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --model_path "./models/fp32/dil_classification/nli/adapter_mask_forget_sup_2000_$id" \
    --approach bert_adapter_mask_forget_sup_ncl \
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
    --save_model
done
#--nepochs 1
#    --train_batch_size 128 \
