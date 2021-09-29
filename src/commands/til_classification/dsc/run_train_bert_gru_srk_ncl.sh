#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_bert_gru_srk_3-%j.out
#SBATCH --gres gpu:1

for id in 3
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id,full \
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --approach bert_gru_srk_ncl \
    --train_batch_size 200 \
    --eval_batch_size 200
done
#--nepochs 1
#    --train_data_size 200 \
