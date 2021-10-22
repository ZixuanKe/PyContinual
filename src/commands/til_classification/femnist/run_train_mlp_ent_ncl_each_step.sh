#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1


for id in 4
do
     python  run.py \
    --note random$id,6200,online,each_step \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_ncl \
    --eval_each_step \
    --ent_id \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_ncl_$id"
done

# always set eval size to 1

#--nepochs 1
#    --train_data_size 500
