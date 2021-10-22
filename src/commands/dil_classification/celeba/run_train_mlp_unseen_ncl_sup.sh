#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 2:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    python  run.py \
    --note random$id,sup,head,norm,unseen,100\
    --ntasks 10 \
    --task celeba \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_ncl  \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --unseen \
    --resume_from_file "./models/fp32/dil_classification/celeba/mlp_ncl_sup_head_norm$id" \
    --resume_from_aux_file "./models/fp32/dil_classification/celeba/mlp_ncl_sup_head_norm$id" \
    --eval_only \
    --eval_batch_size 128
done
#--nepochs 1
#    --train_data_size 500
