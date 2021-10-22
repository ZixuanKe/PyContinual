#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1

for id in 1 2 3 4
do
     python  run.py \
    --note random$id,20tasks,online \
    --ntasks 20 \
    --task cifar100 \
    --scenario til_classification \
    --idrandom $id \
    --approach cnn_ewc_ncl \
    --ent_id \
    --model_path "./models/fp32/til_classification/cifar100/cnn_ewc_20_$id" \
    --resume_from_file "./models/fp32/til_classification/cifar100/cnn_ewc_20_$id"
done
#--nepochs 1


#nepochs=100,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,lamb=5000
