#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_w2v_ewc-%j.out
#SBATCH --gres gpu:1

for id in 0 1 2 3 4
#for id in 0
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment w2v \
    --note random$id,full \
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach w2v_kim_ewc_ncl \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000 \
    --lamb 5000
done
#--nepochs 1
#    --train_data_size 200 \
