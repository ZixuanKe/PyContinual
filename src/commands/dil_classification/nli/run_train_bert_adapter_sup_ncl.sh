#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_nli_adapter_sup_ncl_1000-%j.out
#SBATCH --gres gpu:1


for id in 0
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,sup,1000\
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --train_data_size 1000 \
    --dev_data_size 1000 \
    --temp 1 \
    --base_temp 1 \
    --sup_loss
done
#--nepochs 1
