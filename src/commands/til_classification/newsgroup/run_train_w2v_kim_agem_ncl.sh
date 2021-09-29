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
    --approach w2v_kim_a-gem_ncl \
    --note random$id \
    --ntasks 10 \
    --class_per_task 2 \
    --task newsgroup \
    --idrandom $id \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --lr=0.05 \
    --lr_min=1e-4 \
    --lr_factor=3 \
    --lr_patience=3 \
    --clipgrad=10000 \
    --nepochs 100 \
    --buffer_size 128 \
    --buffer_percent 0.02 \
    --gamma 0.5
done
#--nepochs 1
