#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_dsc_capsule_mask_imp_route_share_full_0-%j.out
#SBATCH --gres gpu:1

for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,transfer_route,share,rlarger_share,full\
    --ntasks 10 \
    --exp 2layer_whole \
    --idrandom $id \
    --nclasses 2 \
    --scenario dil_classification \
    --output_dir './OutputBert' \
    --model_path "./models/fp32/dil_classification/dsc/capsule_mask_imp_route_share_ent_full_$id" \
    --approach bert_adapter_capsule_mask_ent_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --use_imp \
    --task dsc \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 20 \
    --transfer_route \
    --share_conv \
    --larger_as_share \
    --save_model \
    --resume_model \
    --resume_from_file "./models/fp32/dil_classification/dsc/capsule_mask_imp_route_share_ent_full_$id" \
    --resume_from_task 8
done
#    --train_data_size 500 \
#--nepochs 1
#    --train_batch_size 10 \
#    --learning_rate 3e-4

#    --apply_one_layer_shared for 0 is bad
