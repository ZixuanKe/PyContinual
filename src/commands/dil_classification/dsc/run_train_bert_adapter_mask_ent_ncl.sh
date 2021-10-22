#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o dil_dsc_adapter_mask_ent_full_0-%j.out
#SBATCH --gres gpu:1


for id in 0
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,full \
    --ntasks 10 \
    --nclasses 2 \
    --task dsc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --model_path "./models/fp32/dil_classification/dsc/adapter_mask_ent_full_$id" \
    --approach bert_adapter_mask_ent_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 20 \
    --save_model
done

#    --train_data_size 500
#    --train_data_size 2000 \
