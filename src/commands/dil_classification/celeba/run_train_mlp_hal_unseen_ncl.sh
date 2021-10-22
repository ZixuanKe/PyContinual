#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,100,unseen \
    --ntasks 10 \
    --task celeba \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_hal_ncl \
    --unseen \
    --resume_from_file "./models/fp32/dil_classification/celeba/mlp_hal_$id" \
    --resume_from_aux_file "./models/fp32/dil_classification/celeba/mlp_hal_$id" \
    --eval_only \
    --eval_batch_size 128
done


#128 is the total maximum batch size