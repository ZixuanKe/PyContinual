#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,last_id \
    --task mnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_cat_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
	--last_id \
    --model_path "./models/fp32/dil_classification/mnist/mlp_cat_$id" \
    --save_model
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size