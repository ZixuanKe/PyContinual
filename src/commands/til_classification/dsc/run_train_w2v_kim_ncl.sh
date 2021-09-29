#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_dsc_w2v_ncl-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
    python  run.py \
    --experiment w2v \
    --note random$id,full \
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --scenario til_classification \
    --approach w2v_kim_ncl \
    --eval_batch_size 128 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000
done

    #--nepochs 1
