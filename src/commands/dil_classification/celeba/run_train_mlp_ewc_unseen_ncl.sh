#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_1-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
    python  run.py \
    --note random$id,100,unseen \
    --ntasks 10 \
    --task celeba \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_ewc_ncl\
    --unseen \
    --resume_from_file "./models/fp32/dil_classification/celeba/mlp_ewc_$id" \
    --resume_from_aux_file "./models/fp32/dil_classification/celeba/mlp_ewc_$id" \
    --eval_only \
    --eval_batch_size 128
done
#--nepochs 1


#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,lamb=5000
