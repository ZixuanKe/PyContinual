#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_dsc_bert_adapter_capsule_mask_ent_online_-%j.out
#SBATCH --gres gpu:1


for id in 0 2 3 4 1
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,online,full \
    --ntasks 10 \
    --nclasses 2 \
    --task dsc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ent_ncl \
    --model_path "./models/fp32/dil_classification/dsc/capsule_mask_ent_full_$id" \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --eval_batch_size 1 \
    --num_train_epochs 20 \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/dil_classification/dsc/capsule_mask_ent_full_$id" \
    --resume_from_task 9
done

#    --train_data_size 2000 \
#    --train_data_size 2000 \
