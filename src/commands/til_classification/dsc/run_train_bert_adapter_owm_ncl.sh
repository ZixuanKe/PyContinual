#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_dsc_bert_adapter_owm_4%j.out
#SBATCH --gres gpu:1


for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,full \
    --ntasks 10 \
    --task dsc \
    --tasknum 10 \
    --idrandom $id \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_owm_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 20 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_owm  \
    --bert_adapter_size 50
done
#--nepochs 1
#    --train_data_size 200 \
