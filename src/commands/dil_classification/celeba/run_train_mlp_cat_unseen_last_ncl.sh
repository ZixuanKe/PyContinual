#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,100,unseen,last_id \
    --task celeba \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_cat_ncl \
    --unseen \
    --last_id \
    --model_path "./models/fp32/dil_classification/celeba/mlp_cat_$id" \
    --eval_only \
    --resume_from_file "./models/fp32/dil_classification/celeba/mlp_cat_$id" \
    --eval_batch_size 128
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size