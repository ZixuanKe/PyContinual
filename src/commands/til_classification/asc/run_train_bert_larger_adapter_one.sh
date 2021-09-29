#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_dsc_larger_adapter_one_0-%j.out
#SBATCH --gres gpu:1

#export TRANSFORMERS_CACHE='/HPS/MultiClassSampling/work/zixuan/model_cache'
#export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 3 4
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-large-uncased' \
    --note random$id,large\
    --ntasks 19 \
    --task asc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --approach bert_adapter_one \
    --experiment bert \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --learning_rate 1e-5 \
    --bert_hidden_size 1024 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter
done
#--nepochs 1
#-beta 0.03 --ratio 0.5 --lr_rho 0.001 --alpha 0.01
#    --learning_rate 1e-5 \

#    --ratio 0.125 \
#    --beta 0.002 \
#    --lr_rho 5 \


#    --ratio 0.125 \
#    --beta 0.002 \
#    --lr_rho 0.01 \
#    --alpha 5 \
# --train_data_size 200 \
