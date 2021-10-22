#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
     python run.py \
    --note random$id,online \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_owm_ncl \
    --ent_id \
    --model_path "./models/fp32/til_classification/femnist/mlp_owm_ncl_$id" \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_owm_ncl_$id"
done
#--nepochs 1
#lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,