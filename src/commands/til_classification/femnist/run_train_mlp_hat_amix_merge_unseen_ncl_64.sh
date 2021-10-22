#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_1-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,6200,64ep,unseen\
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_hat_merge_ncl \
    --unseen \
    --ent_id \
    --resume_from_file "./models/fp32/til_classification/femnist/mlp_tnc64_$id" \
    --resume_from_aux_file "./models/fp32/til_classification/femnist/mlp_aux_tnc64_$id"
done
#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2

#    --lr 0.05 \
#    --lr_min 1e-4 \
#    --lr_factor 3 \
#    --lr_patience 5 \
#    --clipgrad 10000

#semantic cap size 1000, 500, 2048

