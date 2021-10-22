#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_asc_kim_acl-%j.out
#SBATCH --gres gpu:1


for id in  0
do
     python run.py \
    --note random0,std$id \
    --task femnist \
    --scenario til_classification \
    --idrandom 0 \
    --approach mlp_acl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/til_classification/femnist/mlp_acl_std$id" \
    --save_model\
    --seed $id
done
