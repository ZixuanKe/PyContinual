#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_adapter_agem_0-%j.out
#SBATCH --gres gpu:1


for id in 0
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --output_dir './OutputBert' \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --idrandom $id \
    --scenario til_classification \
    --approach bert_adapter_a-gem_ncl \
    --experiment bert \
    --apply_bert_attention_output \
    --apply_bert_output \
    --build_adapter \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --buffer_size 128 \
    --buffer_percent 0.02 \
    --gamma 0.5
done
#--nepochs 1
#    --learning_rate 3e-4 \
#    --train_batch_size 200 \
#    --train_data_size 200 \
