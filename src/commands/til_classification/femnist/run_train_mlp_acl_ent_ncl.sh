#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 8:00:00
#SBATCH -o til_asc_kim_acl-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
     python run.py \
    --note random$id \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_acl_ncl \
    --ent_id \
    --model_path "./models/fp32/til_classification/femnist/mlp_acl_$id" \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_acl_$id"
done
