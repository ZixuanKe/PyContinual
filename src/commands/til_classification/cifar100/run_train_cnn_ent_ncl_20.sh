#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    python  run.py \
    --note random$id,20tasks,online\
    --ntasks 20 \
    --task cifar100 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_ncl \
    --image_size 32 \
    --image_channel 3 \
    --nepochs 1000\
    --ent_id \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/cifar100/cnn_ncl_20_$id" \
    --resume_from_aux_file "./models/fp32/til_classification/cifar100/cnn_ncl_20_$id" \
    --resume_from_task 19 \
    --eval_batch_size 1
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