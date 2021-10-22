#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_acl-%j.out
#SBATCH --gres gpu:1


for id in  4
do
     python run.py \
    --note random$id,0.02 \
    --ntasks 10 \
    --task mnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_acl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --image_size 28 \
    --image_channel 1 \
    --nepochs=1000 \
    --buffer_size 128 \
    --buffer_percent 0.02 \
    --alpha 0.5 \
    --beta 0.5 \
    --model_path "./models/fp32/til_classification/mnist/mlp_acl_$id" \
    --save_model
done
