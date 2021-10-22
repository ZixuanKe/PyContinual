#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_nli_ncl_4-%j.out
#SBATCH --gres gpu:1


#NCL
for id in 4
do
    python  run.py \
    --note random$id\
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_ncl  \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --exit_after_first_task\
    --print_report
done
#--nepochs 1
#    --train_data_size 500


#NCL-SUP
for id in 4
do
    python  run.py \
    --note random$id\
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_ncl  \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --exit_after_first_task\
    --print_report
done
#--nepochs 1
#    --train_data_size 500


#EWC
for id in 4
do
     python  run.py \
    --note random$id \
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_ewc_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --lamb=500\
    --exit_after_first_task\
    --print_report
done
#--nepochs 1


#UCL

for id in 4
do
    python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id \
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_ucl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --exit_after_first_task\
    --print_report
done
#--nepochs 1


#OWM
for id in 4
do
    python run.py \
    --note random$id \
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_owm_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --exit_after_first_task\
    --print_report
done

#derpp++
for id in  4
do
     python run.py \
    --note random$id \
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_derpp_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --exit_after_first_task\
    --print_report
done


#hal
for id in  4
do
     python run.py \
    --note random$id\
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_hal_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --exit_after_first_task\
    --print_report
done

#acl
for id in  4
do
    python run.py \
    --note random$id \
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_acl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --last_id \
    --exit_after_first_task\
    --print_report
done


#cat
for id in 4
do
     python run.py \
    --note random$id \
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_cat_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
	--last_id\
    --exit_after_first_task\
    --print_report
done



#hat
for id in 4
do
     python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id\
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_hat_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --last_id\
    --exit_after_first_task\
    --print_report
done


#tnc
for id in 4
do
    python run.py \
    --note random$id\
    --task cifar100 \
    --ntasks 10 \
    --ntasks 20 \
    --scenario dil_classification \
    --approach cnn_hat_merge_ncl \
    --idrandom $id \
    --eval_batch_size 128 \
    --train_batch_size 128\
    --exit_after_first_task \
    --print_report
done

for id in  4
do
    python run.py \
    --note random0\
    --task cifar100 \
    --ntasks 20 \
    --scenario dil_classification \
    --idrandom $id \
    --approach cnn_hat_merge_ncl \
    --sup_loss \
    --sup_head \
    --sup_head_norm \
    --eval_batch_size 128 \
    --train_batch_size 128\
    --exit_after_first_task \
    --print_report
done
