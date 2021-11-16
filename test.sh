#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_large_capsule_mask_imp_route_share_list_0-%j.out
#SBATCH --gres gpu:1

export TRANSFORMERS_CACHE='./model_cache'
export HF_DATASETS_CACHE='./dataset_cache'
#


for id in 0
do
    python  run.py \
		--bert_model 'bert-base-uncased' \
		--backbone bert_adapter \
		--baseline b-cl\
		--task asc \
		--eval_batch_size 128 \
		--train_batch_size 32 \
		--scenario til_classification \
		--idrandom 0  \
		--use_predefine_args
done
#    --model_path "./models/fp32/til_classification/dsc/capsule_mask_imp_route_share_$id" \
#    --save_model
#--nepochs 1
#    --train_batch_size 10 \
#    --learning_rate 3e-4
#    --train_data_size 500 \
#    --apply_one_layer_shared for 0 is bad
