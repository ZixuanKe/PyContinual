#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    python  run.py \
    --note random$id,100\
    --ntasks 10 \
    --task celeba \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_one  \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/til_classification/celeba/mlp_one_$id" \
    --save_model
done
#--nepochs 1
#    --train_data_size 500
