#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_adapter_l2_0-%j.out
#SBATCH --gres gpu:1


for id in 4
do
     python  run.py \
    --bert_model 'bert-base-uncased' \
    --output_dir './OutputBert' \
    --note random$id,full\
    --ntasks 10 \
    --idrandom $id \
    --scenario til_classification \
    --task dsc \
    --approach bert_adapter_l2_ncl \
    --experiment bert_adapter \
    --apply_bert_attention_output \
    --apply_bert_output \
    --build_adapter \
    --eval_batch_size 200 \
    --num_train_epochs 20 \
    --lamb 0.5
done
#--nepochs 1
#    --learning_rate 3e-4 \
#    --train_batch_size 200 \
#    --train_data_size 200 \
