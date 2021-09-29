#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_newsgroup_adapter_l2_4-%j.out
#SBATCH --gres gpu:1

export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --output_dir './OutputBert' \
    --note random$id\
    --idrandom $id \
    --ntasks 10 \
    --scenario til_classification \
    --class_per_task 2 \
    --task newsgroup \
    --approach bert_adapter_l2_ncl \
    --experiment bert_adapter \
    --apply_bert_attention_output \
    --apply_bert_output \
    --build_adapter \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --lamb 0.5
done
#--nepochs 1
#    --learning_rate 3e-4 \
#    --train_batch_size 200 \
