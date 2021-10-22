#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    python run.py \
    --note random$id,online \
    --ntasks 10 \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_hal_ncl \
    --ent_id \
    --model_path "./models/fp32/til_classification/femnist/mlp_hal_$id" \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_hal_$id"
done


#128 is the total maximum batch size