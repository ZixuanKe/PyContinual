#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1

for id in 4
do
     python  run.py \
    --note random$id,6200,unseen,each_step \
    --task femnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_ewc_ncl \
    --unseen \
    --eval_each_step \
    --resume_from_file "./models/fp32/dil_classification/femnist/mlp_ewc_$id" \
    --eval_only \
    --eval_batch_size 128
done
#--nepochs 1


#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,lamb=5000
