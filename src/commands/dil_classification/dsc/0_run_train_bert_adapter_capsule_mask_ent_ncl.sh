#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_dsc_capsule_mask_200_1-%j.out
#SBATCH --gres gpu:1

for id in 1
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,200\
    --ntasks 10 \
    --nclasses 2 \
    --scenario dil_classification \
    --task dsc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --model_path "./models/fp32/dil_classification/dsc/capsule_mask_ent_200_$id" \
    --approach bert_adapter_capsule_mask_ent_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --eval_batch_size 128 \
    --num_train_epochs 20 \
    --train_data_size 200 \
    --save_model
done

#    --train_batch_size 14 \
#    --apply_one_layer_shared for 1,0,3
#    --apply_two_layer_shared for 2,4
#    --train_data_size 500 \
