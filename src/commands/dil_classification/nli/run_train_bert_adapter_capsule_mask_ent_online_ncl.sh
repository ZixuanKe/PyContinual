#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o dil_nli_capsule_mask_0-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,online,2000 \
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ent_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --eval_batch_size 1 \
    --num_train_epochs 10 \
    --train_data_size 2000 \
    --dev_data_size 2000 \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/dil_classification/nli/capsule_mask_ent_2000_$id" \
    --resume_from_task 4
done

#    --train_batch_size 14 \
#    --apply_one_layer_shared for 1,0,3
#    --apply_two_layer_shared for 2,4
#    --train_data_size 500 \
#    --resume_model \
#    --resume_from_file "./models/fp32/dil_classification/nli/capsule_mask_ent_$id" \
#    --resume_from_task 3