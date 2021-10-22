#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_kim_gem_0-%j.out
#SBATCH --gres gpu:1

#TODO: GEM baseline, could be time-consuming, consider A-GEM
#TODO： need to change...
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_a-gem_ncl \
    --note random$id \
    --idrandom $id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --nepochs 100 \
    --buffer_size 128 \
    --buffer_percent 0.02 \
    --gamma 0.5 \
    --lr 0.2

done
#--nepochs 1
