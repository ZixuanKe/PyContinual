#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_cifar10_acl_-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    python run.py \
    --note random$id,last_id,online \
    --ntasks 10 \
    --nclasses 10 \
    --task cifar10 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_acl_ncl \
    --ent_id \
    --model_path "./models/fp32/dil_classification/cifar10/cnn_acl_10_$id" \
    --resume_from_file "./models/fp32/dil_classification/cifar10/cnn_acl_10_$id"
done
