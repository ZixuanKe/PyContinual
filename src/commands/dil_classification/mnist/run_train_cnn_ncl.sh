#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


for id in 0
do
    CUDAA_VISIBLE_DEVICES=1 python  run.py \
    --note random$id,sup,300\
    --ntasks 10 \
    --nclasses 62 \
    --task femnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --data_size full \
    --train_data_size 300 \
    --image_size 28 \
    --image_channel 1 \
    --nepoch 100 \
    --sup_loss
done
#--nepochs 1
#    --train_data_size 500
