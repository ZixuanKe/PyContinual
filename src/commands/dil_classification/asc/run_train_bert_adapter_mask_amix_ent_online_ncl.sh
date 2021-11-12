#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_0-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
     python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,classic,ent_id\
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --idrandom $id \
    --approach bert_adapter_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --eval_batch_size 1 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --temp 1 \
    --base_temp 1 \
    --output_dir './OutputBert' \
    --augment_distill \
    --distill_head \
    --sup_loss \
    --amix \
    --ent_id \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/dil_classification/asc/adapter_mask_amix_ent_$id" \
    --resume_from_task 18
done
#    --use_dissimilar \
#    --model_path "./models/fp32/dil_classification/asc/adapter_mask_amix_dis_ent_$id" \

#    --augment_distill \
#    --distill_head \
#--share_gate
#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2
#    --pooled_rep_contrast \
#    --l2_norm \

#    --amix_head
#    --sup_head \
