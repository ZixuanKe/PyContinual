#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 35:00:00
#SBATCH -o til_dsc_back-TSM_-%j.out
#SBATCH --gres gpu:1

for id in 1
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,transfer_route,share,full\
    --ntasks 10 \
    --exp 2layer_whole \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule  \
    --apply_two_layer_shared \
    --use_imp \
    --use_gelu \
    --task dsc \
    --scenario til_classification \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 20 \
    --transfer_route \
    --share_conv \
    --model_path "./models/fp32/til_classification/dsc/capsule_full_$id" \
    --save_model
done
#    --train_data_size 200 \
