#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_newsgroup_adapter_gem_4-%j.out
#SBATCH --gres gpu:1

#export HF_DATASETS_CACHE='/HPS/MultiClassSampling/work/zixuan/dataset_cache'

for id in 1 2 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --output_dir './OutputBert' \
    --note random$id,0.05,incase\
    --ntasks 10 \
    --scenario til_classification \
    --class_per_task 2 \
    --task newsgroup \
    --idrandom $id \
    --approach bert_adapter_gem_ncl \
    --experiment bert_adapter \
    --apply_bert_attention_output \
    --apply_bert_output \
    --build_adapter \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --gamma 0.5
#    --model_path "./models/fp32/til_classification/asc/adapter_gem_0.05_$id" \
#    --save_model \
#    --resume_model \
#    --resume_from_file "./models/fp32/til_classification/asc/adapter_gem_0.05_$id" \
#    --resume_from_task 7
done
#--nepochs 1
#    --learning_rate 3e-4 \
#    --train_batch_size 200 \
