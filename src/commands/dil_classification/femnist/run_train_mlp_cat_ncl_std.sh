#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in 0
do
     python run.py \
    --note random0,std$id,last_id,6200 \
    --ntasks 10 \
    --task femnist \
    --scenario dil_classification \
    --idrandom 0 \
    --approach mlp_cat_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
	--last_id\
    --model_path "./models/fp32/dil_classification/femnist/mlp_cat_std$id" \
    --save_model\
    --seed $id
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size