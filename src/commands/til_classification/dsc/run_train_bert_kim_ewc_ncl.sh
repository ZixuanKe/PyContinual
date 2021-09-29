#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_dsc_bert_kim_ewc_full_0-%j.out
#SBATCH --gres gpu:1

for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_ewc_ncl \
    --note random$id,full \
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --scenario til_classification \
    --train_batch_size 200 \
    --eval_batch_size 200
done
#--nepochs 1
#    --train_batch_size 200 \
