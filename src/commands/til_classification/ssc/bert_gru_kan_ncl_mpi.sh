#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id \
    --ntasks 10 \
    --task dsc \
    --train_data_size 200 \
    --eval_batch_size 64 \
    --train_batch_size 64 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_gru_kan_ncl
done
#--nepochs 1
