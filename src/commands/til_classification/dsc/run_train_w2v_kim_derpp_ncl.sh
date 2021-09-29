#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_w2v_derpp_4-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
     python  run.py \
    --experiment w2v \
    --note random$id,200 \
    --idrandom $id \
    --ntasks 10 \
    --task dsc \
    --scenario til_classification \
    --approach w2v_kim_derpp_ncl \
    --train_data_size 200 \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000 \
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --alpha 0.5 \
    --beta 0.5
done

    #--nepochs 1
