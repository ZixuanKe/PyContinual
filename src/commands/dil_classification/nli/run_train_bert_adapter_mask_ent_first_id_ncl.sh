#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 3:00:00
#SBATCH -o dsc_nli_first_id_-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,first_id \
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mask_ent_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --first_id \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/dil_classification/nli/adapter_mask_ent_$id" \
    --resume_from_task 4
done