#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1


for id in 4
do
     python  run.py \
    --note random$id\
    --ntasks 10 \
    --task mnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --image_size 28 \
    --image_channel 1 \
    --nepochs 300 \
    --model_path "./models/fp32/til_classification/mnist/mlp_ncl_$id" \
    --save_model
done
#--nepochs 1
#    --train_data_size 500
