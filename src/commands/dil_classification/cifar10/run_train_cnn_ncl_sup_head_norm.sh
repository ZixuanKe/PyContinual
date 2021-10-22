#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3
do
    python  run.py \
    --note random$id,sup,head,norm\
    --ntasks 10 \
    --task cifar10 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_ncl \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/dil_classification/cifar10/cnn_ncl_sup_head_norm$id" \
    --resume_model \
    --resume_from_file "./models/fp32/dil_classification/cifar10/cnn_ncl_sup_head_norm$id" \
    --resume_from_task 9 \
    --eval_only

done
#--nepochs 1
#    --train_data_size 500
#    --model_path "./models/fp32/dil_classification/cifar10/cnn_ncl_sup_head_norm$id" \
#    --save_model