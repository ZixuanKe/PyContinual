#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_newsgroup_capsule_mask_imp_route_share_0-%j.out
#SBATCH --gres gpu:1

export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 0
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,transfer_route,share,rlarger_list\
    --ntasks 10 \
    --exp 2layer_whole \
    --idrandom $id \
    --class_per_task 2 \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --use_imp \
    --task newsgroup \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --transfer_route \
    --share_conv \
    --larger_as_list \
    --model_path "./models/fp32/til_classification/newsgroup/capsule_mask_imp_route_list_$id" \
    --save_model \
    --resume_model \
    --resume_from_file "./models/fp32/til_classification/newsgroup/capsule_mask_imp_route_list_$id" \
    --resume_from_task 6
done

#--nepochs 1
#    --train_batch_size 10 \
#    --learning_rate 3e-4
#    --train_data_size 500 \
#    --apply_one_layer_shared for 0 is bad
