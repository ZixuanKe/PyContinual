#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_nli_bert_adapter_mask_sup_trans_1-%j.out
#SBATCH --gres gpu:1


for id in 1
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --model_path "./models/fp32/dil_classification/nli/adapter_mask_sup_trans_full_$id" \
    --approach bert_adapter_mask_trans_sup_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 10
    --save_model
done
#--nepochs 1
