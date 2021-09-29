#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o slurm-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_ucl_ncl \
    --experiment bert \
    --train_data_size 200 \
    --eval_batch_size 32 \
    --train_batch_size 28 \
    --num_train_epochs 20 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_ucl  \
    --ratio 0.125 \
    --beta 0.002 \
    --tasknum 19 \
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