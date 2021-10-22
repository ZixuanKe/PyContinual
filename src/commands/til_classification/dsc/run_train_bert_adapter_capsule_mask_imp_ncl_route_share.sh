#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_capsule_mask_imp_route_share_0-%j.out
#SBATCH --gres gpu:1

for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,transfer_route,share,rlarger_share,one_layer,adapter768\
    --ntasks 10 \
    --exp 2layer_whole \
    --idrandom $id \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_one_layer_shared \
    --use_imp \
    --task dsc \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 20 \
    --transfer_route \
    --share_conv \
    --larger_as_share \
    --train_data_size 200 \
    --adapter_size 768
done
#    --model_path "./models/fp32/til_classification/dsc/capsule_mask_imp_route_share_$id" \
#    --save_model
#--nepochs 1
#    --train_batch_size 10 \
#    --learning_rate 3e-4
#    --train_data_size 500 \
#    --apply_one_layer_shared for 0 is bad
