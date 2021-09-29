#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 35:00:00
#SBATCH -o til_dsc_back-TRKSM_-%j.out
#SBATCH --gres gpu:1

for id in 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,2layer_whole,full\
    --ntasks 10 \
    --task newsgroup \
    --exp 2layer_whole \
    --idrandom $id \
    --output_dir './OutputBert' \
    --class_per_task 2 \
    --scenario til_classification \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --use_imp
done

#--nepochs 1
#    --apply_one_layer_shared for 1,0,3
#    --apply_two_layer_shared for 2,4
