#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_bert_kim_ucl_full-4%j.out
#SBATCH --gres gpu:1

for id in 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id,full \
    --ntasks 10 \
    --tasknum 10 \
    --task dsc \
    --idrandom $id \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach bert_kim_ucl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --nepochs 50 \
    --lr=0.01 \
    --ratio 0.125 \
    --beta 0.002 \
    --lr_rho 0.01 \
    --alpha 5 \
    --optimizer SGD \
    --clipgrad 100 \
    --lr_min 2e-6 \
    --lr_factor 3 \
    --lr_patience 5
done
#--nepochs 1
#    --train_data_size 200 \
