#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_1-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
     python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,20tasks,online\
    --ntasks 20 \
    --task cifar10 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_hat_ncl \
    --nepochs 1000 \
    --image_size 32 \
    --image_channel 3 \
    --ent_id \
    --model_path "./models/fp32/til_classification/cifar10/cnn_hat_ncl_20_$id" \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/cifar10/cnn_hat_ncl_20_$id" \
    --resume_from_task 19 \
    --eval_batch_size 1
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

#    --model_path "./models/fp32/dil_classification/celeba/cnn_hat_amix_ent_$id" \
#    --save_model