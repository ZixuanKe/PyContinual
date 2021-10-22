#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    python run.py \
    --note random0,seed$id,online \
    --ntasks 10 \
    --task femnist \
    --scenario til_classification \
    --idrandom 0 \
    --approach mlp_derpp_ncl \
    --ent_id \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_derpp_ncl_std$id" \
    --seed $id
done
