#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_1-%j.out
#SBATCH --gres gpu:1


for id in 0
do
    CUDA_VISIBLE_DEVICES=1 python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,Attn-HCHP,sup,naug1,selfattn,amix,sup,true_id\
    --ntasks 10 \
    --nclasses 62 \
    --task femnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_hat_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --data_size small \
    --nepochs 300 \
    --image_size 28 \
    --image_channel 1 \
    --temp 1 \
    --base_temp 1 \
    --output_dir './OutputBert' \
    --amix \
    --attn_type self \
    --task_based \
    --mix_type Attn-HCHP-Outside \
    --naug 1 \
    --true_id
done
#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2
#    --augment_distill \
#    --distill_head \
#    --lr 0.05 \
#    --lr_min 1e-4 \
#    --lr_factor 3 \
#    --lr_patience 5 \
#    --clipgrad 10000

#semantic cap size 1000, 500, 2048

#    --model_path "./models/fp32/dil_classification/celeba/cnn_hat_amix_ent_$id" \
#    --save_model