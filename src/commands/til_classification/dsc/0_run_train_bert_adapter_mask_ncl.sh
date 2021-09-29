#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_adapter_mask_full_0-%j.out
#SBATCH --gres gpu:1


for id in 0
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,full,adapter768 \
    --ntasks 10 \
    --task dsc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --adapter_size 768 \
    --eval_batch_size 128 \
    --num_train_epochs 20
done

#    --train_data_size 500
#    --model_path "./models/fp32/til_classification/dsc/adapter_mask_full_$id" \
