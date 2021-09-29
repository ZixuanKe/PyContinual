#!/bin/bash
#!/home/zixuan/anaconda3/envs/pytorch/bin/python

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

#for id in 0 1 2 3 4
for id in 1
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,epoch24\
    --ntasks 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_gen_hat_ncl \
    --experiment bert_gen_hat \
#    --activate_layer_num -3
done
#--nepochs 1
