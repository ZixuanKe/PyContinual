#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id \
    --ntasks 10 \
    --nclasses 10 \
    --task mnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_derpp_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --image_size 28 \
    --image_channel 1 \
    --nepochs=1000 \
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --alpha 0.5 \
    --beta 0.5
done
