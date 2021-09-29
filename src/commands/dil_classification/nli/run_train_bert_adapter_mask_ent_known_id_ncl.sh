#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o slurm_-%j.out
#SBATCH --gres gpu:1
# no need to train, one can use the trained model directly

for id in 3
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,known_id \
    --ntasks 5 \
    --nclasses 3 \
    --task nli \
    --scenario dil_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_mask_ent_ncl \
    --experiment bert \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_mask \
    --eval_batch_size 128 \
    --num_train_epochs 10 \
    --known_id \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/dil_classification/nli/adapter_mask_ent_$id" \
    --resume_from_task 4
done

#    --train_data_size 500 \
