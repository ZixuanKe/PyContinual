#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 4
do
    python  run.py \
    --note random$id,sup,head,norm,20tasks\
    --ntasks 20 \
    --task cifar100 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_ncl \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/dil_classification/cifar100/cnn_ncl_20_sup_head_norm$id" \
    --save_model
done
#--nepochs 1
#    --train_data_size 500
#    --model_path "./models/fp32/dil_classification/cifar10/cnn_ncl_sup_head_norm$id" \
#    --save_model