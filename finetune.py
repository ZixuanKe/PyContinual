#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.


#TODO: config + CL + tensorboard
#TODO: please consider FSDP: https://huggingface.co/docs/accelerate/fsdp


import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import datasets
import transformers
from accelerate import Accelerator
# from accelerate.logging import get_logger
from accelerate.utils import set_seed
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    PretrainedConfig,
    get_scheduler,
)
# from transformers.utils import get_full_repo_name, is_offline_mode
from transformers.utils.versions import require_version
import config
import utils
from dataloader.data import get_dataset
from datasets import Dataset, DatasetDict, concatenate_datasets


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


args = config.parse_args()
args = utils.model.prepare_sequence_finetune(args)

from approaches.finetune import Appr
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
# torch.autograd.set_detect_anomaly(True)

# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
# If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
# in the environment

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision=args.mixed_precision,fp16=args.fp16, kwargs_handlers=[ddp_kwargs])

# accelerator = (
#     Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
# )
if args.source_prefix is None and args.model_name_or_path in [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
]:
    logger.warning(
        "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
        "`--source_prefix 'summarize: ' `"
    )
# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)


# Handle the repository creation
if accelerator.is_main_process:
    if args.push_to_hub:
        pass
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

accelerator.wait_for_everyone()


# Load pretrained model and tokenizer
#
# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
if args.config_name:
    config = AutoConfig.from_pretrained(args.config_name)
elif args.model_name_or_path:
    config = AutoConfig.from_pretrained(args.model_name_or_path)
else:
    config = CONFIG_MAPPING[args.model_type]()
    logger.warning("You are instantiating a new config instance from scratch.")

if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
elif args.base_model_name_or_path: # for training only

    add_space_tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=not args.use_slow_tokenizer,
                                              add_prefix_space=True)
    normal_tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=not args.use_slow_tokenizer)

    if args.task_name in args.ner_datasets:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=not args.use_slow_tokenizer, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=not args.use_slow_tokenizer)

else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )

args.tokenizer = tokenizer
args.add_space_tokenizer = add_space_tokenizer
args.normal_tokenizer = normal_tokenizer


datasets,taskcla = get_dataset(accelerator=accelerator, logger=logger, args=args)
# print('datasets: ',datasets)
model = utils.model.lookfor_model_finetune(taskcla,args,config)

model.model.resize_token_embeddings(len(args.tokenizer))
if model.teacher is not None:
    model.teacher.resize_token_embeddings(len(args.tokenizer))

args.config = utils.model.deepcopy(config)
args.taskcla = taskcla



if 'bart' in args.model_name_or_path and config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

logger.info('==> Preparing data..')


if 'mtl' in args.baseline or 'comb' in args.baseline:

    for t in range(args.ft_task + 1):
        if t == 0:
            train_dataset = datasets[t]['train']

        else:
            train_dataset = concatenate_datasets([train_dataset, datasets[t]['train']])

else:
    # TODO: we may want to save some previous data
    train_dataset = datasets[args.ft_task]['train']
    dev_dataset = datasets[args.ft_task]['dev']

##




# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.



text_column = 'source'
summary_column = 'target'
column_names = [text_column,summary_column]


# ======================================================================
ner_features = train_dataset.features

# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

if args.task_name in args.ner_datasets: #place holder
    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.}
    # label_list = get_label_list(train_dataset[summary_column])
    label_list = utils.data.label_list_dict[args.task_name]
    label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)


    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
            # Reorganize `label_list` to match the ordering of the model.
            label_list = [model.config.id2label[i] for i in range(num_labels)]
            label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels:"
                f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
            )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}  # TODO: careful if you want to cut the dataset

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Preprocessing the datasets.
    # First we tokenize all the texts.

    # Tokenize all texts and align the labels with them.

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            utils.data.tokenize_and_align_labels,
            fn_kwargs={
                'label_to_id':label_to_id,
                'b_to_i_label':b_to_i_label,
                'text_column': text_column,
                'summary_column': summary_column,
                'eval_t': None,
                'args': args},

            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
        
        
        
        dev_dataset = dev_dataset.map(
            utils.data.tokenize_and_align_labels,
            fn_kwargs={
                'label_to_id':label_to_id,
                'b_to_i_label':b_to_i_label,
                'text_column': text_column,
                'summary_column': summary_column,
                'eval_t': None,
                'args': args},

            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

else:

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            utils.data.preprocess_function,
            fn_kwargs = {
                'text_column':text_column,
                'summary_column':summary_column,
                'args':args},
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        dev_dataset = dev_dataset.map(
            utils.data.preprocess_function,
            fn_kwargs={
                'text_column':text_column,
                'summary_column':summary_column,
                'args':args},
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )


print('taskcla: ',taskcla)

# Log a few random samples from the training set:

for index in random.sample(range(len(train_dataset)), 1):

    label = train_dataset[index]['labels']

    if args.pad_to_max_length:
        label = [l for l in label if l != -100]

    logger.info(
        f"Sample {index} of the training set: {train_dataset[index]}. Decode to: {args.tokenizer.decode(train_dataset[index]['input_ids'])} and {args.tokenizer.decode(label)}")

data_collator = utils.data.MyDataCollatorForSeq2Seq(
    args.tokenizer,
    model=model,
    pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    mlm_probability=args.mlm_probability
    # important, you cannot set n_tokens to a random number then
)
# this gives us two different input and label


print('train_dataset: ',len(train_dataset))

train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
dev_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
train_pool_loader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.replay_sample_per_task)

# above train/dev done ======================================
# bellow is tesitng, invovle different type of datasets and need different hyper-poarameters


dev_loaders = []
# for eval_t in range(args.ft_task + 1): # last one is the current one
for eval_t in range(args.ft_task + 1):  # last one is the current one

    dev_dataset = datasets[eval_t]['dev']
    args.task_name = args.all_tasks[eval_t]  # self.args.task_name has chaned
    print('args.task_name : ',args.task_name)
    with accelerator.main_process_first():

        if args.task_name in args.ner_datasets:
            dev_dataset = dev_dataset.map(
                utils.data.tokenize_and_align_labels,
                fn_kwargs={
                'label_to_id':label_to_id,
                'b_to_i_label':b_to_i_label,
                'text_column': text_column,
                'summary_column': summary_column,
                'eval_t': eval_t,
                'args': args
                },
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            dev_dataset = dev_dataset.map(
                utils.data.preprocess_function,
                fn_kwargs={
                'text_column':text_column,
                'summary_column':summary_column,
                'args':args},
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )


    data_collator = utils.data.MyDataCollatorForSeq2Seq(
        args.tokenizer,
        model=model,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        mlm_probability=args.mlm_probability
        # important, you cannot set n_tokens to a random number then
    )

    dev_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    dev_loaders.append(dev_dataloader)


#TODO: temporaty changed
test_loaders = []
# for eval_t in range(args.ft_task + 1): # last one is the current one
for eval_t in range(args.ft_task + 1):  # last one is the current one

    test_dataset = datasets[eval_t]['test']
    args.task_name = args.all_tasks[eval_t]  # self.args.task_name has chaned
    print('args.task_name : ',args.task_name)
    with accelerator.main_process_first():

        if args.task_name in args.ner_datasets:
            test_dataset = test_dataset.map(
                utils.data.tokenize_and_align_labels,
                fn_kwargs={
                'label_to_id':label_to_id,
                'b_to_i_label':b_to_i_label,
                'text_column': text_column,
                'summary_column': summary_column,
                'eval_t': eval_t,
                'args': args
                },
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            test_dataset = test_dataset.map(
                utils.data.preprocess_function,
                fn_kwargs={
                'text_column':text_column,
                'summary_column':summary_column,
                'args':args},
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )


    data_collator = utils.data.MyDataCollatorForSeq2Seq(
        args.tokenizer,
        model=model,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        mlm_probability=args.mlm_probability
        # important, you cannot set n_tokens to a random number then
    )

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_loaders.append(test_dataloader)


#recover
args.task_name = args.all_tasks[args.ft_task]  # self.args.task_name has chaned

appr = Appr(config, args)
appr.train(model,train_dataloader,train_dataset,dev_loaders,test_loaders,train_pool_loader,accelerator)

