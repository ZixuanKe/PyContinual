#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0
do
    python run.py \
    --note random0,seed$id \
    --task femnist \
    --scenario til_classification \
    --idrandom 0 \
    --approach mlp_cat_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/til_classification/femnist/mlp_cat_std$id" \
    --save_model\
    --seed $id
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size