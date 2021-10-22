#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    python run.py \
    --note random$id,0.02 \
    --ntasks 10 \
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_hal_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --image_size 32 \
    --image_channel 3 \
    --nepochs=1000 \
    --buffer_size 128 \
    --buffer_percent 0.02 \
    --gamma 0.1 \
    --beta 0.5 \
    --hal_lambda 0.1\
    --model_path "./models/fp32/til_classification/cifar10/cnn_hal0.02_$id" \
    --save_model
done


#128 is the total maximum batch size