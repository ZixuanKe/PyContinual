#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --note random$id,sup,head,norm \
    --ntasks 10 \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_one \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --model_path "./models/fp32/til_classification/femnist/mlp_one_sup_head_norm$id" \
    --save_model
done
#--nepochs 1
#    --train_data_size 500
