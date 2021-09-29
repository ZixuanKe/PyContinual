#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_1-%j.out
#SBATCH --gres gpu:1


for id in 1
do
     python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,amix,Attn-HCHP-Outside,distill_head,sup,augment_distill,naug1\
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --idrandom $id \
    --approach bert_adapter_mask_aux_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --temp 1 \
    --base_temp 1 \
    --output_dir './OutputBert' \
    --two_stage \
    --aux_net \
    --augment_distill \
    --distill_head \
    --sup_loss \
    --amix \
    --mix_type Attn-HCHP-Outside \
    --attn_type cos \
    --semantic_cap_size 768 \
    --naug 0
done

#    --tmix \
#    --ntmix 1 \
#    --alpha 16 \
#
#--share_gate
#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2
#    --pooled_rep_contrast \
#    --l2_norm \

#temp, ntix