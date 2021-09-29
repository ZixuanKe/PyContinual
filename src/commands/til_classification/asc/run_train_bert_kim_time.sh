#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_bert_kim_ncl-4%j.out
#SBATCH --gres gpu:1






#KAN

for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random2 \
    --ntasks 19 \
    --scenario til_classification \
    --task asc \
    --idrandom $id  \
    --output_dir './OutputBert' \
    --approach bert_gru_kan_ncl \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task
done


#SRK

for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random2 \
    --ntasks 19 \
    --scenario til_classification \
    --task asc \
    --idrandom $id  \
    --output_dir './OutputBert' \
    --approach bert_gru_srk_ncl \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task
done


#EWC
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_ewc_ncl \
    --note random$id \
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 200 \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task
done




#UCL
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_ucl_ncl \
    --note random$id \
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 200 \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task
done

#OWM
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_owm_ncl \
    --note random$id \
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 200 \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task
done



#HAT
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_hat_ncl \
    --note random$id \
    --ntasks 19 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 200 \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task
done






#L2
#for id in 4
#do
#    CUDA_VISIBLE_DEVICES=0 python  run.py \
#    --bert_model 'bert-base-uncased' \
#    --experiment bert \
#    --approach bert_kim_l2_ncl \
#    --note random$id \
#    --ntasks 19 \
#    --task asc \
#    --scenario til_classification \
#    --idrandom $id \
#    --output_dir './OutputBert' \
#    --eval_batch_size 200 \
#    --train_batch_size 128 \
#    --nepochs 1 \
#    --exit_after_first_task
#done
#--nepochs 1

#AGEM
#for id in 4
#do
#    python  run.py \
#    --bert_model 'bert-base-uncased' \
#    --experiment bert \
#    --approach bert_kim_a-gem_ncl \
#    --note random$id \
#    --ntasks 19 \
#    --task asc \
#    --idrandom $id \
#    --output_dir './OutputBert' \
#    --scenario til_classification \
#    --eval_batch_size 128 \
#    --train_batch_size 128 \
#    --nepochs=100 \
#    --lr=0.05 \
#    --lr_min=1e-4 \
#    --lr_factor=3 \
#    --lr_patience=3 \
#    --clipgrad=10000\
#    --buffer_size 128 \
#    --buffer_percent 0.05 \
#    --gamma 0.5 \
#    --nepochs 1 \
#    --exit_after_first_task
#done
#--nepochs 1


#derpp
#
#for id in 4
#do
#     python run.py \
#    --bert_model 'bert-base-uncased' \
#    --experiment bert \
#    --approach bert_kim_derpp_ncl \
#    --note random$id\
#    --ntasks 19 \
#    --task asc \
#    --scenario til_classification \
#    --idrandom $id \
#    --output_dir './OutputBert' \
#    --eval_batch_size 256 \
#    --train_batch_size 128 \
#    --nepochs=100 \
#    --lr=0.05 \
#    --lr_min=1e-4 \
#    --lr_factor=3 \
#    --lr_patience=3 \
#    --clipgrad=10000 \
#    --buffer_size 128 \
#    --buffer_percent 0.05 \
#    --alpha 0.5 \
#    --beta 0.5 \
#    --nepochs 1 \
#    --exit_after_first_task
#done
