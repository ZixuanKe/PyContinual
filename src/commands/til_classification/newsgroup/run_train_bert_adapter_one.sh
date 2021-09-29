#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_newsgroup_adapter_ncl-%j.out
#SBATCH --gres gpu:1

export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 0 1 2 3 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --experiment bert \
    --ntasks 10 \
    --scenario til_classification \
    --class_per_task 2 \
    --task newsgroup \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_one \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter
done
#--nepochs 1
#-beta 0.03 --ratio 0.5 --lr_rho 0.001 --alpha 0.01

#    --ratio 0.125 \
#    --beta 0.002 \
#    --lr_rho 5 \


#    --ratio 0.125 \
#    --beta 0.002 \
#    --lr_rho 0.01 \
#    --alpha 5 \