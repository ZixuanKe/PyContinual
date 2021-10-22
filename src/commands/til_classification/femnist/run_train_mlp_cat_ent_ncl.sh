#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  4
do
    python run.py \
    --note random$id,online \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_cat_ncl \
    --ent_id \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_cat_$id"
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size

