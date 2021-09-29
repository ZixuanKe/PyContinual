#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_ncl_full_4-%j.out
#SBATCH --gres gpu:1

for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,full\
    --ntasks 10 \
    --task dsc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --num_train_epochs 20
done
#--nepochs 1
#    --train_data_size 500
#    --nclasses 3 \
