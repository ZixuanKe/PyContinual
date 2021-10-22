#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
     python  run.py \
    --note random$id,6200,sup,head,norm\
    --ntasks 10 \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_ncl \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --model_path "./models/fp32/til_classification/femnist/mlp_ncl_sup_head_norm$id" \
    --save_model
done
#--nepochs 1
#    --train_data_size 500
