#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1
do
    python run.py \
    --note random$id,ent_id,6200,unseen \
    --ntasks 10 \
    --task femnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_cat_ncl \
    --unseen \
    --ent_id \
    --model_path "./models/fp32/dil_classification/femnist/mlp_cat_$id" \
    --resume_from_file "./models/fp32/dil_classification/femnist/mlp_cat_$id"
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size