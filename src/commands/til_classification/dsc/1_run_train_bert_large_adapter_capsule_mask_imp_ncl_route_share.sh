#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o til_dsc_large_capsule_mask_imp_route_share_1-%j.out
#SBATCH --gres gpu:1

export TRANSFORMERS_CACHE='/HPS/MultiClassSampling/work/zixuan/model_cache'
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 1
do
    python  run.py \
    --bert_model 'bert-large-uncased' \
    --note random$id,imp,transfer_route,share,rlarger_share,large,200\
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
    --train_data_size 200 \
    --train_batch_size 32 \
    --num_train_epochs 20 \
    --transfer_route \
    --share_conv \
    --larger_as_share \
    --bert_hidden_size 1024 \
    --model_path "./models/fp32/til_classification/dsc/capsule_mask_imp_route_share_large_$id" \
    --save_model
done
#    --model_path "./models/fp32/til_classification/dsc/capsule_mask_imp_route_share_$id" \
#    --save_model
#--nepochs 1
#    --train_batch_size 10 \
#    --learning_rate 3e-4
#    --train_data_size 500 \
#    --apply_one_layer_shared for 0 is bad
