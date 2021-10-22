#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,online\
    --ntasks 10 \
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_cat_ncl \
    --image_size 32 \
    --image_channel 3 \
    --nepochs=1000 \
	--n_head 5 \
	--similarity_detection auto \
	--loss_type multi-loss-joint-Tsim \
    --ent_id \
    --model_path "./models/fp32/til_classification/cifar10/cnn_cat_$id" \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/cifar10/cnn_cat_$id" \
    --resume_from_task 9 \
    --eval_batch_size 1
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size