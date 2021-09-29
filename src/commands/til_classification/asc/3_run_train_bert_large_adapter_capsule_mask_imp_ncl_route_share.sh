#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 40:00:00
#SBATCH -o til_asc_large_capsule_mask_imp_route_share_3-%j.out
#SBATCH --gres gpu:1

export TRANSFORMERS_CACHE='/HPS/MultiClassSampling/work/zixuan/model_cache'
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 3
do
    python  run.py \
    --bert_model 'bert-large-uncased' \
    --note random$id,imp,transfer_route,share,rlarger_share,large\
    --ntasks 19 \
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
    --task asc \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --transfer_route \
    --share_conv \
    --larger_as_share \
    --bert_hidden_size 1024 \
    --model_path "./models/fp32/til_classification/asc/capsule_mask_imp_route_share_large_$id" \
    --save_model\
    --resume_model \
    --resume_from_file "./models/fp32/til_classification/asc/capsule_mask_imp_route_share_large_$id" \
    --resume_from_task 11
done
#    --model_path "./models/fp32/til_classification/dsc/capsule_mask_imp_route_share_$id" \
#    --save_model
#--nepochs 1
#    --train_batch_size 10 \
#    --learning_rate 3e-4
#    --train_data_size 500 \
#    --apply_one_layer_shared for 0 is bad
