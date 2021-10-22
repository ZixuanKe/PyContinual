#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,100 \
    --ntasks 10 \
    --task celeba \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_cat_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --train_data_size 100 \
    --data_size full \
    --image_size 32 \
    --image_channel 3 \
    --nepochs=1000 \
	--n_head 5 \
	--similarity_detection auto \
	--loss_type multi-loss-joint-Tsim \
    --model_path "./models/fp32/til_classification/celeba/mlp_cat_$id" \
    --save_model
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size