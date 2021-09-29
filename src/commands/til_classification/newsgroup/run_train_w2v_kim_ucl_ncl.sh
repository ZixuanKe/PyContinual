#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_newsgroup_w2v_ucl-%j.out
#SBATCH --gres gpu:1
export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 0 1 2 3 4
do
     python  run.py \
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
    --approach w2v_kim_ucl_ncl \
    --ratio 0.125 \
    --beta 0.002 \
    --lr_rho 0.01 \
    --alpha 5 \
    --optimizer SGD \
    --clipgrad 100 \
    --lr_min 2e-6 \
    --lr_factor 3 \
    --lr_patience 5
done
#--nepochs 1
# --beta 0.0002 --ratio 0.125 --lr_rho 0.01 --alpha 0.3