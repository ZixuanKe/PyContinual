#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  4
do
    python run.py \
    --note random$id,online,each_step \
    --ntasks 10 \
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_hal_ncl\
    --eval_each_step \
    --ent_id \
    --resume_from_file "./models/fp32/til_classification/cifar10/cnn_hal_$id"
done


#128 is the total maximum batch size