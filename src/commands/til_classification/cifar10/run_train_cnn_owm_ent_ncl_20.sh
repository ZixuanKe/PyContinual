#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
    python run.py \
    --note random$id,20tasks,online \
    --ntasks 20 \
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach cnn_owm_ncl \
    --image_size 32 \
    --image_channel 3 \
    --ent_id \
    --model_path "./models/fp32/til_classification/cifar10/cnn_owm_20_$id" \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/cifar10/cnn_owm_20_$id" \
    --resume_from_task 19 \
    --eval_batch_size 1
    --nepochs=1000
done
#--nepochs 1
#lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,