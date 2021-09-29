#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 1:00:00
#SBATCH -o dsc_asc_last_id_-%j.out
#SBATCH --gres gpu:1


for id in 1
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,last_id,shared-specific,cat,augment_distill,separate \
    --ntasks 19 \
    --nclasses 3 \
    --task asc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
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
    --mix_type shared-specific \
    --last_id \
    --augment_distill \
    --distill_type separate
done

#    --share_gate \
#    --semantic_cap_size 768