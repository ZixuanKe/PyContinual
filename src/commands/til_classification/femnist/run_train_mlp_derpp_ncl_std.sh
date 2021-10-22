#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
     python run.py \
    --note random0,std$id \
    --task femnist \
    --scenario til_classification \
    --idrandom 0 \
    --approach mlp_derpp_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/til_classification/femnist/mlp_derpp_ncl_std$id" \
    --save_model\
    --seed $id
done
