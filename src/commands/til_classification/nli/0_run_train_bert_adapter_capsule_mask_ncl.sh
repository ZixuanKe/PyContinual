#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 30:00:00
#SBATCH -o til_nli_capsule_mask_0-%j.out
#SBATCH --gres gpu:1

for id in 0
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --ntasks 5 \
    --scenario til_classification \
    --task nli \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_mask_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule_mask  \
    --apply_two_layer_shared \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --save_model \
    --model_path "./models/fp32/til_classification/nli/capsule_mask_$id" \
    --save_model\
    --resume_model \
    --resume_from_file "./models/fp32/til_classification/nli/capsule_mask_$id" \
    --resume_from_task 3
done

#    --train_batch_size 14 \
#    --apply_one_layer_shared for 1,0,3
#    --apply_two_layer_shared for 2,4
#    --train_data_size 500 \
#    --train_data_size 2000 \
#    --dev_data_size 2000