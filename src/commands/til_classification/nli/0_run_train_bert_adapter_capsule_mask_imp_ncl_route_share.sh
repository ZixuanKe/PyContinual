#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 35:00:00
#SBATCH -o til_nli_capsule_mask_imp_route_share_0-%j.out
#SBATCH --gres gpu:1

for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,transfer_route,share,rlarger_share\
    --ntasks 5 \
    --task nli \
    --exp 2layer_whole \
    --idrandom $id \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --use_imp \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --transfer_route \
    --share_conv \
    --larger_as_share \
    --model_path "./models/fp32/til_classification/nli/capsule_mask_imp_route_list_1" \
    --save_model \
    --resume_model \
    --resume_from_file "./models/fp32/til_classification/nli/capsule_mask_imp_route_list_$id" \
    --resume_from_task 4
done
#--nepochs 1
#    --train_batch_size 10 \
#    --learning_rate 3e-4
#    --apply_one_layer_shared for 0 is bad

#    --train_data_size 2000 \
#    --dev_data_size 2000