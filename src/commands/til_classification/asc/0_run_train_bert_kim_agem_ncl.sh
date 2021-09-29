#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_gem_0-%j.out
#SBATCH --gres gpu:1

#TODO: GEM baseline, could be time-consuming, consider A-GEM
#TODO： need to change...


for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_a-gem_ncl \
    --note random$id \
    --ntasks 19 \
    --task asc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000\
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --gamma 0.5
done
#--nepochs 1
