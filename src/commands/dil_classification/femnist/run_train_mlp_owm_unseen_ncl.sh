#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python run.py \
    --note random$id,6200,unseen \
    --task femnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_owm_ncl \
    --unseen \
    --model_path "./models/fp32/dil_classification/femnist/mlp_owm_$id" \
    --resume_from_file "./models/fp32/dil_classification/femnist/mlp_owm_$id" \
    --eval_only \
    --eval_batch_size 128
done
#--nepochs 1
#lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,