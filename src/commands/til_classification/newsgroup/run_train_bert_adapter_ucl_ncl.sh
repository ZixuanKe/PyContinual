#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_newsgroup_bert_adapter_kim_ucl_0-%j.out
#SBATCH --gres gpu:1

export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'
for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_ucl_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_ucl  \
    --ratio 0.125 \
    --beta 0.002 \
    --tasknum 10 \
    --alpha 5
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