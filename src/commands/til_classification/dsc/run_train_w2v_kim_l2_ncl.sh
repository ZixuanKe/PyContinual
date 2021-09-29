#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_dsc_kim_gem_4-%j.out
#SBATCH --gres gpu:1

#TODO: GEM baseline, could be time-consuming, consider A-GEM
#TODO： need to change...


for id in 0 1 2 3 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment w2v \
    --approach w2v_kim_l2_ncl \
    --note random$id,200 \
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --train_data_size 200 \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000 \
    --nepochs 100 \
    --lamb 0.5
done
#--nepochs 1
#    --train_data_size 200 \
