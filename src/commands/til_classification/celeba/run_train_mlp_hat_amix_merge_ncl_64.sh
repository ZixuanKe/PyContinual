#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_1-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
     python run.py \
    --note random$id,100,64ep\
    --ntasks 10 \
    --task celeba \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_hat_merge_ncl \
    --eval_batch_size 128 \
    --train_batch_size 64\
    --model_path "./models/fp32/til_classification/celeba/mlp64_tnc_$id" \
    --aux_model_path "./models/fp32/til_classification/celeba/mlp64_aux_tnc_$id" \
    --save_model
done

#ID 2 ---> #BATCH=32
#ID 4 ---> #BATCH=64
#ID 3 ---> #BATCH=64

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

