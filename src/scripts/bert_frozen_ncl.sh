#!/bin/bash




export HF_DATASETS_CACHE='/sdb/zke4/dataset_cache'
export TRANSFORMERS_CACHE='/sdb/zke4/model_cache'

CUDA_VISIBLE_DEVICES=1 python  run.py \
  --bert_model 'bert-base-uncased' \
  --backbone bert_frozen \
  --baseline ncl\
  --task dsc \
  --eval_batch_size 128 \
  --train_batch_size 32 \
  --scenario til_classification \
  --idrandom 1  \
  --use_predefine_args