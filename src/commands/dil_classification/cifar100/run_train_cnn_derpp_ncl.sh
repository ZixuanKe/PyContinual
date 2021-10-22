#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
     python run.py \
    --note random$id,0.02 \
    --ntasks 10 \
    --nclasses 100 \
    --task cifar100 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_derpp_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --image_size 32 \
    --image_channel 3 \
    --nepochs=1000 \
    --buffer_size 128 \
    --buffer_percent 0.02 \
    --alpha 0.5 \
    --beta 0.5
done
