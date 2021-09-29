#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 1:00:00
#SBATCH -o dil_dsc_bert_adapter_mask_ent_unify_-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,unify_detect,full \
    --ntasks 10 \
    --nclasses 2 \
    --task dsc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mask_ent_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 20 \
    --unify_detect \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/dil_classification/dsc/adapter_mask_ent_full_$id" \
    --resume_from_task 9
done

#    --train_data_size 2000 \
