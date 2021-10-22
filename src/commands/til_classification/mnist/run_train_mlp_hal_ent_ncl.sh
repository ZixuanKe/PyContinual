#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o til_asc_kim_derpp_4-%j.out
#SBATCH --gres gpu:1


for id in  0 1 2 3 4
do
    python run.py \
    --note random$id,online \
    --ntasks 10 \
    --task mnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_hal_ncl \
    --train_batch_size 128 \
    --image_size 28 \
    --image_channel 1 \
    --nepochs=1000 \
    --buffer_size 128 \
    --buffer_percent 0.05 \
    --gamma 0.1 \
    --beta 0.5 \
    --hal_lambda 0.1\
    --ent_id \
    --model_path "./models/fp32/til_classification/mnist/mlp_hal_$id" \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/mnist/mlp_hal_$id" \
    --resume_from_task 9 \
    --eval_batch_size 1
done


#128 is the total maximum batch size