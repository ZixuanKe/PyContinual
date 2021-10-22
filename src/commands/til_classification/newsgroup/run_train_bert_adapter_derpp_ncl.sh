#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_dsc_adapter_derpp_0-%j.out
#SBATCH --gres gpu:1


for id in 0
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --idrandom $id \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_derpp_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --eval_batch_size 200 \
    --num_train_epochs 10 \
    --buffer_size 128 \
    --buffer_percent 0.02 \
    --alpha 0.5 \
    --beta 0.5
done
#--nepochs 1
