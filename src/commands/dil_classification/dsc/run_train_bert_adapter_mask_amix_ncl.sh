#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o dil_dsc_adapter_mask_mixup_0-%j.out
#SBATCH --gres gpu:1


for id in 0
do
   python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,classic\
    --ntasks 10 \
    --nclasses 2 \
    --task dsc \
    --scenario dil_classification \
    --idrandom $id \
    --approach bert_adapter_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --eval_batch_size 128 \
    --num_train_epochs 20 \
    --train_data_size 200 \
    --temp 1 \
    --base_temp 1 \
    --output_dir './OutputBert' \
    --augment_distill \
    --distill_head \
    --sup_loss \
    --amix \
    --semantic_cap_size 2048 \
    --last_id
done

#    --tmix \
#    --ntmix 1 \
#    --alpha 16 \

#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2
#    --pooled_rep_contrast \
#    --l2_norm \
#    --alpha 16 \
#    --separate-mix True \
#        --last_id \
#    --e_start_mixup 5 \
#    --build_adapter_mask \
#    --build_adapter \
#    --sup_loss \
#    --distill_head
#    --augment_distill \
#    --distill_type 'separate' \
