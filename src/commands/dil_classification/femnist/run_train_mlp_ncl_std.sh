#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    python  run.py \
    --note random0,std$id,6200\
    --ntasks 10 \
    --task femnist \
    --scenario dil_classification \
    --idrandom 0 \
    --approach mlp_ncl  \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/dil_classification/femnist/mlp_ncl_std_$id" \
    --save_model \
    --seed $id
done
#--nepochs 1
#    --train_data_size 500
