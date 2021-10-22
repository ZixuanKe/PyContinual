#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 6:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1


for id in  4
do
     python run.py \
    --note random$id \
    --ntasks 10 \
    --task mnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_derpp_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --image_size 28 \
    --image_channel 1 \
    --nepochs=1000 \
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --alpha 0.5 \
    --beta 0.5\
    --model_path "./models/fp32/til_classification/mnist/mlp_derpp_ncl_$id" \
    --save_model
done
