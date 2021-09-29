#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_dsc_bert_kim_cat-%j.out
#SBATCH --gres gpu:1

for id in 0
#for id in 0
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id,200 \
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach bert_kim_cat_ncl \
    --train_data_size 200 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
	--n_head 5 \
	--similarity_detection auto \
	--loss_type multi-loss-joint-Tsim \
    --nepochs 100 \
    --lr=0.5 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000
done
#--nepochs 1
# dsc no "w2v_as"
