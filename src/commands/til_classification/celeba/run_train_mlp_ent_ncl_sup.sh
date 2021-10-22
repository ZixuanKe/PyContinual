#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --note random$id,100,sup,head,norm,online \
    --ntasks 10 \
    --task celeba \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_ncl \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --ent_id \
    --model_path "./models/fp32/til_classification/celeba/mlp_ncl_sup_head_norm$id" \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/celeba/mlp_ncl_sup_head_norm$id" \
    --resume_from_task 9 \
    --eval_batch_size 1
done
#--nepochs 1
#    --train_data_size 500
