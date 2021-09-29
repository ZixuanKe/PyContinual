#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_dsc_adapter_mask_mixup_4-%j.out
#SBATCH --gres gpu:1


for id in 4
do
     python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,last_id,transfer_layer_all,augment_trans,pooled_rep_contrast,200\
    --ntasks 10 \
    --nclasses 2 \
    --task dsc \
    --scenario dil_classification \
    --idrandom $id \
    --approach bert_adapter_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 20 \
    --train_data_size 200 \
    --output_dir './OutputBert' \
    --last_id \
    --transfer_layer_all \
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
#    --augment_distill \
#    --tmix \
#    --temp 1 \
#    --base_temp 1 \
#    --alpha 16 \
#    --mix-layers-set 7 9 12 \
#    --separate-mix True \

#    --temp 1 \
#    --base_temp 1 \
#    --tmix \
#    --alpha 16 \
#    --mix-layers-set 7 9 12 \
#    --separate-mix True \
#    --last_id \
#    --augment_distill \
#    --sup_loss

#    --tmix \
#    --temp 1 \
#    --base_temp 1 \
#    --alpha 16 \
#    --mix-layers-set 7 9 12 \
#    --separate-mix True \
#    --augment_trans
