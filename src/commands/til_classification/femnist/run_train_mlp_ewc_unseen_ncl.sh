#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
    python  run.py \
    --note random$id,6200,unseen\
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_ewc_ncl \
    --lamb=5000 \
    --unseen \
    --ent_id \
    --model_path "./models/fp32/til_classification/femnist/mlp_ewc_ncl_$id" \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_ewc_ncl_$id"
done
#--nepochs 1


#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,lamb=5000
