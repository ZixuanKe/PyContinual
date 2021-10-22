#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    python  run.py \
    --note random$id,online\
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_ncl \
    --ent_id \
    --model_path "./models/fp32/til_classification/cifar10/cnn_ncl_$id" \
    --resume_from_file "./models/fp32/til_classification/cifar10/cnn_ncl_$id"
done
#--nepochs 1
#    --train_data_size 500
#lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,
#    --sup_loss

#    --lr 0.05 \
#    --lr_min 1e-4 \
#    --lr_factor 3 \
#    --lr_patience 5 \
#    --clipgrad 10000 \