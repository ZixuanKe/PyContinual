#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_dsc_newsgroup_one-%j.out
#SBATCH --gres gpu:1
#export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 0 1 2 3 4
do
    python  run.py \
    --experiment w2v \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --idrandom $id \
    --scenario til_classification \
    --approach w2v_kim_one \
    --eval_batch_size 128 \
    --nepochs=100 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000
done

    #--nepochs 1
