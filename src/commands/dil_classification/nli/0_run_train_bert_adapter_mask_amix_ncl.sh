#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_nli_adapter_mask_mixup_0-%j.out
#SBATCH --gres gpu:1


for id in 0
do
    python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,amix,Attn-HCHP-Outside,augment_distill,two_stage,2000\
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --approach bert_adapter_mask_aux_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 20 \
    --train_data_size 2000 \
    --dev_data_size 2000 \
    --temp 1 \
    --base_temp 1 \
    --output_dir './OutputBert' \
    --amix \
    --mix_type Attn-HCHP-Outside \
    --n_aug 0 \
    --semantic_cap_size 768 \
    --two_stage \
    --aux_net \
    --augment_distill
done
#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2
#    --pooled_rep_contrast \
#    --l2_norm \