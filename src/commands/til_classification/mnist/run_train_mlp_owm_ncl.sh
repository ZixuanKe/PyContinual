#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1

for id in 4
do
     python run.py \
    --note random$id \
    --ntasks 10 \
    --task mnist \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach mlp_owm_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --image_size 28 \
    --image_channel 1 \
    --nepochs 1000\
    --model_path "./models/fp32/til_classification/mnist/mlp_owm_ncl_$id" \
    --save_model
done
#--nepochs 1
#lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,