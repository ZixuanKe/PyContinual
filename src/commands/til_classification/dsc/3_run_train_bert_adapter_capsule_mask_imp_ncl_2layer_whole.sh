#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 40:00:00
#SBATCH -o til_dsc_back-TRKSM_-%j.out
#SBATCH --gres gpu:1

for id in 3
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,2layer_whole,full\
    --ntasks 10 \
    --task dsc \
    --exp 2layer_whole \
    --idrandom $id \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 20 \
    --use_imp \
    --model_path "./models/fp32/til_classification/dsc/capsule_mask_imp_2lwhole_$id" \
    --save_model
done

#--nepochs 1
#    --apply_one_layer_shared for 1,0,3
#    --apply_two_layer_shared for 2,4
