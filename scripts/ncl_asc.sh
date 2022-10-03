#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o posttrain_procy_qa-%j.out
#SBATCH --gres gpu:4

export HF_DATASETS_CACHE='/sdb/zke4/dataset_cache'
export TRANSFORMERS_CACHE='/sdb/zke4/model_cache'
export TRANSFORMERS_OFFLINE=1


seed=(2021 111 222 333 444 555 666 777 888 999)


for round in 0;
do
  for idrandom in 0;
  do
    for ft_task in $(seq 0 0);
      do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --use_env --master_port 12942 finetune.py \
        --ft_task ${ft_task} \
        --idrandom ${idrandom} \
        --ntasks 19 \
        --baseline 'ncl_asc_bert' \
        --seed ${seed[$round]} \
        --per_device_eval_batch_size 32 \
        --sequence_file 'asc' \
        --per_device_train_batch_size 32 \
        --base_model_name_or_path 'bert-base-uncased' \
        --warmup_ratio 0.5 \
        --patient 50 \
        --num_train_epochs 100 \
        --fp16 \
        --max_source_length 128 \
        --pad_to_max_length
      done
  done
done

# epoch=20 is better for full training
