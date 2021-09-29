#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o til_dsc_bert_kim_ncl-4%j.out
#SBATCH --gres gpu:1

#kAN

for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random2 \
    --scenario til_classification \
    --task dsc \
    --idrandom $id  \
    --output_dir './OutputBert' \
    --approach bert_gru_kan_ncl \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task \
    --train_data_size 200
done




#l2
for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random2 \
    --scenario til_classification \
    --task dsc \
    --idrandom $id  \
    --output_dir './OutputBert' \
    --approach bert_kim_l2_ncl \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task\
    --train_data_size 200
done


#SRK

for id in 4
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --note random2 \
    --scenario til_classification \
    --task dsc \
    --idrandom $id  \
    --output_dir './OutputBert' \
    --approach bert_gru_srk_ncl \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task\
    --train_data_size 200
done


#EWC
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_ewc_ncl \
    --note random$id \
    --task dsc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 200 \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task\
    --train_data_size 200
done




#UCL
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_ucl_ncl \
    --note random$id \
    --task dsc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 200 \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task\
    --train_data_size 200
done

#OWM
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_owm_ncl \
    --note random$id \
    --task dsc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 200 \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task\
    --train_data_size 200
done



#HAT
for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_hat_ncl \
    --note random$id \
    --task dsc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --eval_batch_size 200 \
    --train_batch_size 128 \
    --nepochs 1 \
    --exit_after_first_task\
    --train_data_size 200
done






