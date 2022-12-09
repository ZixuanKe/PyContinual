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
# TODO: please consider FSDP: https://huggingface.co/docs/accelerate/fsdp


from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from approaches.finetune import Appr
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
from datasets import Value
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
from utils import utils
from dataloader.data import get_dataset
from datasets import Dataset, DatasetDict, concatenate_datasets
from networks.baselines import ldbr

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

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
args = utils.prepare_sequence_finetune(args)


# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
# If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
# in the environment

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision=args.mixed_precision,
                          fp16=args.fp16, kwargs_handlers=[ddp_kwargs])

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
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
elif args.base_model_name_or_path:  # for training only

    add_space_tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=not args.use_slow_tokenizer,
                                                        add_prefix_space=True)
    normal_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, use_fast=not args.use_slow_tokenizer)

    if args.task_name in args.ner_datasets:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path, use_fast=not args.use_slow_tokenizer, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path, use_fast=not args.use_slow_tokenizer)

else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )

args.tokenizer = tokenizer
args.add_space_tokenizer = add_space_tokenizer
args.normal_tokenizer = normal_tokenizer


datasets, taskcla = get_dataset(accelerator=accelerator, logger=logger, args=args)
model = utils.lookfor_model_finetune(taskcla, args, config)

model.model.resize_token_embeddings(len(args.tokenizer))
if model.teacher is not None:
    model.teacher.resize_token_embeddings(len(args.tokenizer))

args.config = utils.deepcopy(config)


if 'bart' in args.model_name_or_path and config.decoder_start_token_id is None:
    raise ValueError(
        "Make sure that `config.decoder_start_token_id` is correctly defined")

logger.info('==> Preparing data..')


if 'mtl' in args.baseline or 'comb' in args.baseline:

    for t in range(args.ft_task + 1):
        if t == 0:
            train_dataset = datasets[t]['train']
            dev_dataset = datasets[t]['dev']

        else:
            train_dataset = concatenate_datasets(
                [train_dataset, datasets[t]['train']])
            dev_dataset = concatenate_datasets(
                [dev_dataset, datasets[t]['dev']])

else:
    # TODO: we may want to save some previous data
    train_dataset = datasets[args.ft_task]['train']
    dev_dataset = datasets[args.ft_task]['dev']

if 'ldbr' in args.baseline:
    train_dataset = ldbr.process_dataset(train_dataset, args.tokenizer)
    dev_dataset = ldbr.process_dataset(dev_dataset, args.tokenizer)

def preprocess_function(examples):
    # Temporarily set max_target_length for training.
    padding = "max_length" if args.pad_to_max_length else False

    inputs = examples[text_column]
    targets = examples[summary_column]
    task_id = examples['task']
    if 'cls_labels' in examples:
        cls_labels = examples['cls_labels']

    inputs = [prefix + inp for inp in inputs]

    model_inputs = args.tokenizer(
        inputs, max_length=args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with args.tokenizer.as_target_tokenizer():
        labels = args.tokenizer(
            targets, max_length=args.max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != args.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs['task'] = task_id
    if 'cls_labels' in examples:
        model_inputs['cls_labels'] = cls_labels

    return model_inputs


def tokenize_and_align_labels(examples):

    # Temporarily set max_target_length for training.
    padding = "max_length" if args.pad_to_max_length else False

    tokenized_inputs = args.tokenizer(
        examples[text_column],
        max_length=args.max_length,
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples[summary_column]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                if ('eval_t' not in locals() and 'eval_t' not in globals()) or args.ft_task == 0 or args.ft_task == eval_t:
                    label_ids.append(label_to_id[label[word_idx]])
                else:
                    label_ids.append(
                        label_to_id_dict[args.task_name][label[word_idx]])

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if args.label_all_tokens:
                    label_ids.append(
                        b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["cls_labels"] = labels
    tokenized_inputs["labels"] = labels

    task_id = examples['task']

    inputs = examples[text_column]

    tokenized_inputs['task'] = task_id

    return tokenized_inputs


# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.


prefix = args.source_prefix if args.source_prefix is not None else ""

text_column = 'source'
summary_column = 'target'
column_names = [text_column, summary_column]


label_list_dict = \
    {
        'conll2003': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
        'wnut2017': ['O', 'B-location', 'I-location', 'B-corporation', 'I-corporation', 'B-person', 'I-person',
                     'B-product', 'I-product', 'B-creative-work', 'I-creative-work',
                     'B-group', 'I-group'],
        'wikigold': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
        'ontonote': ['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC',
                     'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE',
                     'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT',
                     'B-EVENT', 'I-EVENT', 'B-WORK_OF_ART', 'I-WORK_OF_ART',
                     'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE',
                     'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME',
                     'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY',
                     'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL',
                     'B-CARDINAL', 'I-CARDINAL'
                     ],
        'btc': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
        'ieer': ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG',
                 'B-PCT', 'I-PCT', 'B-MON', 'I-MON',
                 'B-TIM', 'I-TIM', 'B-DAT', 'I-DAT',
                 'B-DUR', 'I-DUR', 'B-CAR', 'I-CAR',
                 'B-MEA', 'I-MEA'
                 ],
        'ritter': ['O', 'B-person', 'I-person', 'B-geo-loc', 'I-geo-loc', 'B-facility', 'I-facility',
                   'B-company', 'I-company', 'B-sportsteam', 'I-sportsteam',
                   'B-musicartist', 'I-musicartist', 'B-product', 'I-product',
                   'B-tvshow', 'I-tvshow', 'B-movie', 'I-movie',
                   'B-other', 'I-other'
                   ],
        're3d': ['O', 'B-Person', 'I-Person', 'B-DocumentReference', 'I-DocumentReference', 'B-Location', 'I-Location',
                 'B-MilitaryPlatform', 'I-MilitaryPlatform', 'B-Money', 'I-Money',
                 'B-Nationality', 'I-Nationality', 'B-Organisation', 'I-Organisation',
                 'B-Quantity', 'I-Quantity', 'B-Temporal', 'I-Temporal',
                 'B-Weapon', 'I-Weapon'
                 ],
        'gum': ['O', 'B-person', 'I-person', 'B-place', 'I-place', 'B-organization', 'I-organization',
                'B-quantity', 'I-quantity', 'B-time', 'I-time',
                'B-event', 'I-event', 'B-abstract', 'I-abstract',
                'B-substance', 'I-substance', 'B-object', 'I-object',
                'B-animal', 'I-animal', 'B-plant', 'I-plant'
                ]
    }


label_to_id_dict = \
    {
        'conll2003': {l: i for i, l in enumerate(label_list_dict['conll2003'])},
        'wnut2017': {l: i for i, l in enumerate(label_list_dict['wnut2017'])},
        'wikigold': {l: i for i, l in enumerate(label_list_dict['wikigold'])},
        'ontonote': {l: i for i, l in enumerate(label_list_dict['ontonote'])},
        'btc': {l: i for i, l in enumerate(label_list_dict['btc'])},
        'ieer': {l: i for i, l in enumerate(label_list_dict['ieer'])},
        'ritter': {l: i for i, l in enumerate(label_list_dict['ritter'])},
        're3d': {l: i for i, l in enumerate(label_list_dict['re3d'])},
        'gum': {l: i for i, l in enumerate(label_list_dict['gum'])},
    }


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


if args.task_name in args.ner_datasets:  # place holder
    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.}
    # label_list = get_label_list(train_dataset[summary_column])
    label_list = label_list_dict[args.task_name]
    label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    print('label_list: ', label_list)
    print('label_to_id: ', label_to_id)

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
    # TODO: careful if you want to cut the dataset
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

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
            tokenize_and_align_labels,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
        dev_dataset = dev_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

else:

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        dev_dataset = dev_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )


print('taskcla: ', taskcla)

# Log a few random samples from the training set:

for index in random.sample(range(len(train_dataset)), 1):

    label = train_dataset[index]['labels']

    if args.pad_to_max_length:
        label = [l for l in label if l != -100]

    logger.info(
        f"Sample {index} of the training set: {train_dataset[index]}. Decode to: {args.tokenizer.decode(train_dataset[index]['input_ids'])} and {args.tokenizer.decode(label)}")

label_pad_token_id = - \
    100 if args.ignore_pad_token_for_loss else args.tokenizer.pad_token_id

data_collator = DataCollatorForSeq2Seq(
    args.tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    # important, you cannot set n_tokens to a random number then
)


print('train_dataset: ', len(train_dataset))

train_dataloader = DataLoader(train_dataset, shuffle=True,
                              collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
dev_dataloader = DataLoader(
    dev_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
train_pool_loader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                               batch_size=args.per_device_train_pool_batch_size)

# above train/dev done ======================================
# bellow is testing, invovle different type of datasets and need different hyper-parameters


test_loaders = []
for eval_t in range(args.ft_task + 1):  # last one is the current one
    test_dataset = datasets[eval_t]['test']
    if 'ldbr' in args.baseline:
        test_dataset = ldbr.process_dataset(test_dataset, args.tokenizer)
    args.task_name = args.all_tasks[eval_t]  # self.args.task_name has chaned
    if 'mix' in args.baseline and 'pool' in args.baseline:
        args = utils.update_hyparameter_for_mix_pool(
            args)  # update args for the hyper-parameters
    elif 'mix' in args.baseline:
        args = utils.update_hyparameter_for_mix_norm(
            args)  # update args for the hyper-parameters
    print('args.task_name : ', args.task_name)
    with accelerator.main_process_first():
        if args.task_name in args.ner_datasets:
            test_dataset = test_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    data_collator = DataCollatorForSeq2Seq(
        args.tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        # important, you cannot set n_tokens to a random number then
    )

    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_loaders.append(test_dataloader)


appr = Appr(config, args)
appr.train(model, train_dataloader, train_dataset, dev_dataloader,
           test_loaders, train_pool_loader, accelerator)
