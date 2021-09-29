#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 35:00:00
#SBATCH -o til_newsgroup_capsule_mask_full_0-%j.out
#SBATCH --gres gpu:1

for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --model_path "./models/fp32/til_classification/newsgroup/capsule_mask_full_$id" \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --save_model
done

#    --train_batch_size 14 \
#    --apply_one_layer_shared for 1,0,3
#    --apply_two_layer_shared for 2,4
#    --train_data_size 500 \
