#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,100,0.02 \
    --ntasks 10 \
    --nclasses 2 \
    --task celeba \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_hal_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --train_data_size 100 \
    --data_size full \
    --image_size 32 \
    --image_channel 3 \
    --nepochs=1000 \
    --buffer_size 128 \
    --buffer_percent 0.02 \
    --gamma 0.1 \
    --beta 0.5 \
    --hal_lambda 0.1 \
    --model_path "./models/fp32/til_classification/celeba/mlp_hal0.02_ncl_$id" \
    --save_model
done


#128 is the total maximum batch size