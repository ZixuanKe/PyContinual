#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_1-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 4
do
    CUDA_VISIBLE_DEVICES=1 python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,Attn-HCHP,augment_distill,separate,6200\
    --ntasks 10 \
    --nclasses 62 \
    --task femnist \
    --scenario dil_classification \
    --idrandom $id \
    --approach mlp_hat_aux_ncl \
    --eval_batch_size 128 \
    --train_batch_size 32 \
    --aux_nepochs 500 \
    --data_size full \
    --image_size 28 \
    --image_channel 1 \
    --train_data_size 6200 \
    --temp 1 \
    --base_temp 1 \
    --output_dir './OutputBert' \
    --amix \
    --mix_type Attn-HCHP \
    --n_aug 0 \
    --semantic_cap_size 1000 \
    --two_stage \
    --aux_net \
    --augment_distill \
    --distill_type separate
done
#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2
#    --last_id \


#semantic cap size 1000, 500, 2048
#temp: 1, 0.3, 0.07
#nepochs: 100, 300
#hat_as_aug
#n_aug: 2, 0
#nsamples: 8 5

