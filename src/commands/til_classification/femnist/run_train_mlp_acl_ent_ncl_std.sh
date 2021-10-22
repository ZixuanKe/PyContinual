#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 8:00:00
#SBATCH -o til_asc_kim_acl-%j.out
#SBATCH --gres gpu:1


for id in  1 2 3 4 0
do
     python run.py \
    --note random0,seed$id,online \
    --task femnist \
    --scenario til_classification \
    --idrandom 0 \
    --approach mlp_acl_ncl \
    --ent_id \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_acl_std$id" \
    --seed $id

done



#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,lamb=5000
