#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,20tasks \
    --ntasks 20 \
    --task cifar100 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_cat_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/til_classification/cifar100/cnn_cat_20_$id" \
    --resume_from_file "./models/fp32/til_classification/cifar100/cnn_cat_20_$id" \
    --resume_model \
    --resume_from_task 16 \
    --save_model
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size

#13, 17, 17, 17, 16