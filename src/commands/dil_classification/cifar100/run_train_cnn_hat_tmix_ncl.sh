#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_1-%j.out
#SBATCH --gres gpu:1


for id in 0
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,train_twice,transfer_layer,last_id,200\
    --ntasks 10 \
    --nclasses 2 \
    --task celeba \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_hat_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --train_data_size 200 \
    --data_size full \
    --image_size 32 \
    --image_channel 3 \
    --train_twice \
    --last_id \
    --transfer_layer \
    --train_twice
done
#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2


