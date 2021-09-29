#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_bert_gru_kan_4-%j.out
#SBATCH --gres gpu:1

for id in 4
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
    --approach bert_gru_kan_ncl \
    --train_batch_size 128 \
    --eval_batch_size 128
done
#--nepochs 1
#    --train_data_size 200 \
