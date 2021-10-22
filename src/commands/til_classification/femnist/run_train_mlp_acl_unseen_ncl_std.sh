#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 8:00:00
#SBATCH -o til_asc_kim_acl-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
     python run.py \
    --note random0,std$id,unseen \
    --task femnist \
    --scenario til_classification \
    --idrandom 0 \
    --approach mlp_acl_ncl \
    --unseen \
    --ent_id \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_acl_$id"\
    --seed $id
done
