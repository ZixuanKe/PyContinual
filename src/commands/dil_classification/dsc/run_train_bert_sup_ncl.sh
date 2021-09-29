#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 30:00:00
#SBATCH -o dil_dsc_sup_ncl_full-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,temp1,2000\
    --ntasks 10 \
    --nclasses 2 \
    --task dsc \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_sup_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --train_data_size 2000 \
    --num_train_epochs 20 \
    --temp 1 \
    --base_temp 1
done
#--nepochs 1
# --nclasses 2