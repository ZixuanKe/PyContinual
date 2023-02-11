

# PyContinual v1.0.0

## Introduction
We are developing v1.0.0, it improves v0.0 in the followings aspects:

* Improve the readability. We removed most of the unnecessary and duplicated code. We use some more recent packages (Accelerate, Adapter-transformer, Transformer) to help achieve this.  
* We focused on **Transformer-based** CL method. More type of NLP tasks are supported. It currently supports **classification** (ASC, CCD), **extraction** (NER) and **generation** (summerization, dialogue response)
* More underlying LMs are supported. It currently supports **BERT, RoBERTa and BART**
* More recent NLP techniques are supported. It currently supports **adapter and (soft-)prompt**.
* More efficient. Fp16 and multi-node training are supported

## TODOs
We are still working on immigrate v0.0.0 to v1.0.0. At this moment, there are still some TODOs **(PR welcomed)**

* At this moment, it supports the following baselines: NCL, ONE, EWC, HAT, B-CL, CTR, CAT, [SupSup](https://arxiv.org/abs/2006.14769), [L2P](https://arxiv.org/abs/2112.08654), [DER++](https://arxiv.org/abs/2004.07211), [DyTox](https://arxiv.org/abs/2111.11326), [LA-MAML](https://arxiv.org/abs/2007.13904), [MER](https://arxiv.org/abs/1810.11910), [LDBR](https://arxiv.org/abs/2104.05489), DualPrompt, A-GEM
* At this moment, it supports only the Task-incremental scenario (task ID is given in both training and testing)

## Architecture
`./dataloader`: contained dataloader for different data  
`./approaches`: code for training  
`./networks`: code for network architecture  
`./sequence`: different sequences  
`./utils`: common utils and model-specific utils. The default hyper-parameters are hard-coded in utils  
`./tools`: code for pre-processing the data and conduct analysis (e.g. forgetting rate, heatmap...)

## Setup

See ``requirement.txt`` for the required package. Please make sure the you installed the correct version.

## Example


Here is an example:
```python

seed=(2021 111 222 333 444 555 666 777 888 999)

for round in 0;
do
  for idrandom in 0;
  do
    for ft_task in $(seq 0 19);
      do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --use_env --master_port 12942 finetune.py \
        --ft_task ${ft_task} \
        --idrandom ${idrandom} \
        --ntasks 19 \
        --baseline 'adapter_bcl_asc_bert' \
        --seed ${seed[$round]} \
        --per_device_eval_batch_size 32 \
        --sequence_file 'asc' \
        --per_device_train_batch_size 32 \
        --base_model_name_or_path 'bert-base-uncased' \
        --warmup_ratio 0.5 \
        --patient 50 \
        --fp16 \
        --max_source_length 128 \
        --pad_to_max_length \
        --base_dir <your dataset directory>
      done
  done
done

```

Above shows a typical command to run PyContinual v1.0.0 Some of the arguments are easy to understand, We further explain some PyContinual arguments:

  - `idrandom`: which random sequence you want to use  
  - `round`: which seeds you want to use  
  - `base_model_name_or_path` BERT, RoBERTa and BART are supported so far
  - `sequence_file`: see `./sequence` for the supported tasks
  - `baseline`: a string that contains the baseline names, including 
    - one, ncl, prompt_one, prompt_ncl, adapter_one, adapter_ncl, mtl, comb, ewc
    - adapter_bcl, adapter_ctr, adapter_hat, adapter_cat
    - l2p, supsup, derpp, dytox, lamaml, mer, ldbr, dualprompt, agem
  - `base_dir`: you need to make sure your dataset is in this directory 
    
If you have questions about what papers the baseline systems refer to or how to download the data. Please check the [README in Main Branch](https://github.com/ZixuanKe/PyContinual/v0.0.0)

## Contact


Please drop an email to [Zixuan Ke](mailto:zke4@uic.edu), [Yijia Shao](mailto:shaoyj@pku.edu.cn), [Haowei Lin](mailto:linhaowei@pku.edu.cn), or [Xingchang Huang](mailto:huangxch3@gmail.com) if you have any questions regarding to the code. We thank [Bing Liu](https://www.cs.uic.edu/~liub/), [Hu Xu](https://howardhsu.github.io/) and [Lei Shu](https://leishu02.github.io/) for their valuable comments and opinioins.



