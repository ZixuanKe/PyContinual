#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_adapter_ncl-4-%j.out
#SBATCH --gres gpu:1

for id in 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --approach bert_adapter_ncl \
    --experiment bert \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 20 \
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