#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_newsgroup_w2v_owm-%j.out
#SBATCH --gres gpu:1
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'


for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment w2v \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --tasknum 10 \
    --idrandom $id \
    --scenario til_classification \
    --output_dir './OutputBert' \
    --approach w2v_kim_owm_ncl \
    --clipgrad 100 \
    --lr_min 2e-6 \
    --lr_factor 3 \
    --lr_patience 5 \
    --lr 0.05
done
#--nepochs 1