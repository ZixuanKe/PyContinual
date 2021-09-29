#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_adapter_l2_0-%j.out
#SBATCH --gres gpu:1


for id in 0
do
    CUDA_VISIBLE_DEVICES=1 python  run.py \
    --bert_model 'bert-base-uncased' \
    --output_dir './OutputBert' \
    --note random$id\
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --approach bert_adapter_l2_ncl \
    --experiment bert_adapter \
    --apply_bert_attention_output \
    --apply_bert_output \
    --build_adapter \
    --eval_batch_size 200 \
    --xusemeval_num_train_epochs 10 \
    --bingdomains_num_train_epochs 30 \
    --bingdomains_num_train_epochs_multiplier 3 \
    --lamb 0.5
done
#--nepochs 1
#    --learning_rate 3e-4 \
#    --train_batch_size 200 \
