#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_adapter_ncl-4-%j.out
#SBATCH --gres gpu:1




for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,imp,transfer_route,share,rlarger_share,adapter768\
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
    --apply_two_layer_shared \
    --use_imp \
    --task asc \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 1 \
    --xusemeval_num_train_epochs 1 \
    --bingdomains_num_train_epochs 1 \
    --bingdomains_num_train_epochs_multiplier 1 \
    --transfer_route \
    --share_conv \
    --larger_as_share \
    --bert_adapter_size 768\
    --exit_after_first_task
done

#OWM
for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_owm_ncl \
    --experiment bert \
    --num_train_epochs 1 \
    --xusemeval_num_train_epochs 1 \
    --bingdomains_num_train_epochs 1 \
    --bingdomains_num_train_epochs_multiplier 1 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_owm  \
    --bert_adapter_size 50 \
    --exit_after_first_task
done
#--nepochs 1

#ucl
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_ucl_ncl \
    --experiment bert_adapter \
    --eval_batch_size 32 \
    --num_train_epochs 1 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --xusemeval_num_train_epochs 1 \
    --bingdomains_num_train_epochs 1 \
    --bingdomains_num_train_epochs_multiplier 1\
    --exit_after_first_task
done
--nepochs 1






#l2
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_l2_ncl \
    --experiment bert_adapter \
    --eval_batch_size 32 \
    --num_train_epochs 1 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --xusemeval_num_train_epochs 1 \
    --bingdomains_num_train_epochs 1 \
    --bingdomains_num_train_epochs_multiplier 1\
    --exit_after_first_task
done
#--nepochs 1




#EWC
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_ewc_ncl \
    --experiment bert_adapter \
    --eval_batch_size 32 \
    --num_train_epochs 1 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --xusemeval_num_train_epochs 1 \
    --bingdomains_num_train_epochs 1 \
    --bingdomains_num_train_epochs_multiplier 1\
    --exit_after_first_task
done
#--nepochs 1



#HAT
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mask_ncl \
    --experiment bert_adapter \
    --eval_batch_size 32 \
    --num_train_epochs 1 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --xusemeval_num_train_epochs 1 \
    --bingdomains_num_train_epochs 1 \
    --bingdomains_num_train_epochs_multiplier 1\
    --exit_after_first_task
done
#--nepochs 1
