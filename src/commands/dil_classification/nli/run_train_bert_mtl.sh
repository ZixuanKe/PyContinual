#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_nli_mtl_4-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random$id \
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_mtl \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/dil_classification/nli/mtl_$id" \
    --resume_from_task 4
done
#--nepochs 1
#    --train_data_size 500 \
#    --save_model
#    --save_model
#    --model_path "./models/fp32/dil_classification/nli/mtl_$id" \
