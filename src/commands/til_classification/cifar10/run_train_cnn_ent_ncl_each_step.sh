#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  1
do
    python  run.py \
    --note random$id,online,each_step\
    --ntasks 10 \
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_ncl \
    --eval_each_step \
    --ent_id \
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