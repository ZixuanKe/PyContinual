#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0
do
    python run.py \
    --note random0,std$id,unseen \
    --task femnist \
    --scenario til_classification \
    --idrandom 0\
    --approach mlp_cat_ncl \
    --unseen \
    --ent_id \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_cat_std$id"\
    --seed $id
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size