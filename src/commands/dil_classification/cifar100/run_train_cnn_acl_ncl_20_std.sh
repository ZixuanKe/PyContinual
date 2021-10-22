#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_cifar100_acl_-%j.out
#SBATCH --gres gpu:1


for id in  0
do
     python run.py \
    --note random0,std$id,20tasks \
    --ntasks 20 \
    --task cifar100 \
    --scenario dil_classification \
    --idrandom 0 \
    --approach cnn_acl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/dil_classification/cifar100/cnn_acl_20_std$id" \
    --last_id \
    --save_model\
    --seed $id
done
