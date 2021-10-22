#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1


for id in 3 4
do
     python  run.py \
    --note random$id,6200,sup,head,norm,unseen\
    --ntasks 10 \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_ncl \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --unseen \
    --ent_id \
    --model_path "./models/fp32/til_classification/femnist/mlp_ncl_sup_head_norm$id" \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_ncl_sup_head_norm$id"
done
#--nepochs 1
#    --train_data_size 500
