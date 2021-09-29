#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_ncl_full_4-%j.out
#SBATCH --gres gpu:1

for id in 1
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --experiment w2v \
    --note random$id \
    --ntasks 19 \
    --idrandom $id \
    --approach w2v_kim_ncl \
    --eval_batch_size 32 \
    --train_batch_size 128 \
    --task asc \
    --scenario til_classification \
    --exit_after_first_task \
    --nepochs 1
done

    #--nepochs 1
