#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 4
do
    python  run.py \
    --note random$id,sup,head,norm,6200\
    --ntasks 10 \
    --task mnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_ncl  \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/dil_classification/mnist/mlp_ncl_sup_head_norm$id" \
    --save_model
done
#--nepochs 1
#    --train_data_size 500
