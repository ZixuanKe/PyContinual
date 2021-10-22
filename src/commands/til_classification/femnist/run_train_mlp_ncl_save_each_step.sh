#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1


for id in  2 3 4
do
     python  run.py \
    --note random$id,6200\
    --ntasks 10 \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --save_each_step \
    --data_size full \
    --image_size 28 \
    --image_channel 1 \
    --train_data_size 6200 \
    --nepochs 1000 \
    --model_path "./models/fp32/til_classification/femnist/mlp_ncl_$id" \
    --save_model
done
#--nepochs 1
#    --train_data_size 500
