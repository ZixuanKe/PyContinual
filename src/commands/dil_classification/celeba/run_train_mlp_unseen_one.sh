#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    python  run.py \
    --note random$id,100,unseen\
    --ntasks 10 \
    --task celeba \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_one  \
    --unseen \
    --resume_from_file "./models/fp32/til_classification/celeba/mlp_one_$id" \
    --resume_from_aux_file "./models/fp32/til_classification/celeba/mlp_one_$id" \
    --eval_only \
    --eval_batch_size 128
done
#--nepochs 1
#    --train_data_size 500
