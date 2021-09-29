#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_dsc_adapter_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --idrandom $id \
    --note random$id,full \
    --ntasks 10 \
    --task dsc \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach bert_adapter_derpp_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --eval_batch_size 200 \
    --num_train_epochs 20 \
    --buffer_size 128 \
    --buffer_percent 0.02 \
    --gamma 0.5
done
#--nepochs 1
