#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
do
    python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,100 \
    --ntasks 10 \
    --task celeba \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_ucl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128\
    --model_path "./models/fp32/dil_classification/celeba/mlp_ucl_$id" \
    --save_model
done
#--nepochs 1
# lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,