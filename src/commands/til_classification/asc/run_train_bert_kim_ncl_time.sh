#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_bert_kim_ncl-4%j.out
#SBATCH --gres gpu:1

for id in 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_ncl \
    --note random$id \
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 256 \
    --train_batch_size 128 \
    --nepochs=1 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000\
    --exit_after_first_task
done
#--nepochs 1
