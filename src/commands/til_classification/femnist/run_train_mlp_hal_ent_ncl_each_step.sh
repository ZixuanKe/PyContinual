#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in 3
do
    python run.py \
    --note random$id,online,each_step \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_hal_ncl \
    --eval_each_step \
    --ent_id \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_hal_$id"
done


#128 is the total maximum batch size