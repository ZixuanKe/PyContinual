#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  4
do
    python run.py \
    --note random0,std$id,online,20tasks \
    --ntasks 20 \
    --task cifar100 \
    --scenario dil_classification \
    --idrandom 0 \
    --approach cnn_cat_ncl \
    --ent_id \
    --resume_from_file "./models/fp32/dil_classification/cifar100/cnn_cat_20_std$id"\
    --seed $id
done

#TODO: cat's hyper-parameters?

#128 is the total maximum batch size