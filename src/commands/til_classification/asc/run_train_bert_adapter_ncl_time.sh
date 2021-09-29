#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_adapter_ncl-4-%j.out
#SBATCH --gres gpu:1

for id in 0
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_ncl \
    --experiment bert_adapter \
    --eval_batch_size 32 \
    --num_train_epochs 1 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter \
    --xusemeval_num_train_epochs 1 \
    --bingdomains_num_train_epochs 1 \
    --bingdomains_num_train_epochs_multiplier 1\
    --exit_after_first_task
done
#--nepochs 1
