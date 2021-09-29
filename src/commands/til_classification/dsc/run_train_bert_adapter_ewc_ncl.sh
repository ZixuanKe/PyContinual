#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_dsc_bert_adapter_ewc_3-%j.out
#SBATCH --gres gpu:1


for id in 3
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --output_dir './OutputBert' \
    --note random$id,full\
    --ntasks 10 \
    --idrandom $id \
    --scenario til_classification \
    --approach bert_adapter_ewc_ncl \
    --experiment bert_adapter \
    --apply_bert_attention_output \
    --apply_bert_output \
    --build_adapter \
    --task dsc \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --num_train_epochs 20 \
    --lamb 5000
done
#--nepochs 1
#    --learning_rate 3e-4 \
#    --train_data_size 200 \
