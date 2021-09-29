#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_2-%j.out
#SBATCH --gres gpu:1


for id in 2
do
     python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,last_id,augment_trans,pooled_rep_contrast\
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --idrandom $id \
    --approach bert_adapter_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --output_dir './OutputBert' \
    --last_id \
    --tmix \
    --temp 1 \
    --base_temp 1 \
    --alpha 16 \
    --mix-layers-set 7 9 12 \
    --separate-mix True \
    --augment_trans \
    --pooled_rep_contrast
done
#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2
#    --pooled_rep_contrast \
#    --l2_norm \