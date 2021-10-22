#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --note random$id,sup,head,norm,online \
    --ntasks 10 \
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_ncl \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --ent_id \
    --model_path "./models/fp32/til_classification/cifar10/cnn_ncl_sup_head_norm$id" \
    --resume_from_file "./models/fp32/til_classification/cifar10/cnn_ncl_sup_head_norm$id"
done
#--nepochs 1
#    --train_data_size 500
