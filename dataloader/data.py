
import json
import os.path
import random

import jsonlines
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import json
from collections import defaultdict
from tqdm import tqdm
summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "cqasumm": ("source", "target"),
    "nyt": ("source", "target"),
    "ami": ("source", "target"),
}

label2word = {'+': 'great', '-': 'terrible', 'positive': 'great', 'negative': 'terrible', 'neutral': 'okay'}
label2idx = {'+': 0, '-': 1, 'positive': 0, 'negative': 1, 'neutral': 2}




def preprocess(text):
    #replace multiple space by single space.
    text = re.sub("[ ]+"," ",text)
    return text


def get_dataset(accelerator, logger, args):
    asc_datasets = [
        args.base_dir + '/data_cpt/SemEval14-res',
        args.base_dir + '/data_cpt/SemEval14-laptop',

        args.base_dir + '/data_cpt/absa/dat/Bing3Domains/Speaker',
        args.base_dir + '/data_cpt/absa/dat/Bing3Domains/Router',
        args.base_dir + '/data_cpt/absa/dat/Bing3Domains/Computer',

        args.base_dir + '/data_cpt/absa/dat/Bing5Domains/Nokia6610',
        args.base_dir + '/data_cpt/absa/dat/Bing5Domains/NikonCoolpix4300',
        args.base_dir + '/data_cpt/absa/dat/Bing5Domains/CreativeLabsNomadJukeboxZenXtra40GB',
        args.base_dir + '/data_cpt/absa/dat/Bing5Domains/CanonG3',
        args.base_dir + '/data_cpt/absa/dat/Bing5Domains/ApexAD2600Progressive',

        args.base_dir + '/data_cpt/absa/dat/Bing9Domains/CanonPowerShotSD500',
        args.base_dir + '/data_cpt/absa/dat/Bing9Domains/CanonS100',
        args.base_dir + '/data_cpt/absa/dat/Bing9Domains/DiaperChamp',
        args.base_dir + '/data_cpt/absa/dat/Bing9Domains/HitachiRouter',
        args.base_dir + '/data_cpt/absa/dat/Bing9Domains/ipod',
        args.base_dir + '/data_cpt/absa/dat/Bing9Domains/LinksysRouter',
        args.base_dir + '/data_cpt/absa/dat/Bing9Domains/MicroMP3',
        args.base_dir + '/data_cpt/absa/dat/Bing9Domains/Nokia6600',
        args.base_dir + '/data_cpt/absa/dat/Bing9Domains/Norton',
    ]


    taskcla = []
    f_name = './sequence/' + args.sequence_file
    data = {}

    # Others
    # f_name = dataset_name + '_random_' + str(args.ntasks)
    with open(f_name, 'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    for t in range(args.ntasks):
        dataset_name = random_sep[t]

        data[t] = {}
        data[t]['name'] = dataset_name

        print('dataset_name: ', dataset_name)


        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.

        if dataset_name in args.asc_datasets:
            datasets = {}
            dataset_path = asc_datasets[args.asc_datasets.index(dataset_name)]

            if 'Bing' in dataset_path:
                data[t]['ncla'] = 2
            elif 'SemEval' in dataset_path:
                data[t]['ncla'] = 3
            taskcla.append((t, int(data[t]['ncla'])))


            for ds in ['train', 'test', 'dev']:  # all test

                datasets[ds] = {}
                datasets[ds]['source'] = []
                datasets[ds]['target'] = []
                datasets[ds]['task'] = []
                datasets[ds]['cls_labels'] = []


                with open(dataset_path + '/{}.json'.format(ds), 'r') as f:
                    tmp_data = json.load(f)
                for dt in tmp_data:
                    label = label2word[tmp_data[dt]['polarity']]
                    sentence = tmp_data[dt]['sentence']
                    term = tmp_data[dt]['term']
                    cls_labels = label2idx[tmp_data[dt]['polarity']]
                    datasets[ds]['source'].append(sentence + args.tokenizer.sep_token + term)
                    datasets[ds]['target'].append(label)
                    datasets[ds]['task'].append(t)
                    datasets[ds]['cls_labels'].append(cls_labels)


            print('datasets["train"]: ', len(datasets["train"]['task']))

            raw_datasets = DatasetDict({
                'train': Dataset.from_dict(datasets["train"]),
                'validation': Dataset.from_dict(datasets["dev"]),
                'test': Dataset.from_dict(datasets["test"]),
            })

        elif dataset_name in args.five_large_datasets:
            path_dict = {'yahoo': args.base_dir +'/data_cpt/yahoo_answers_csv/',
                         'yelp': args.base_dir + '/data_cpt/yelp_review_polarity_csv/',
                         'amazon': args.base_dir + '/data_cpt/amazon_review_polarity_csv/',
                         'dbpedia': args.base_dir + '/data_cpt/dbpedia_csv/',
                         'agnews': args.base_dir + '/data_cpt/ag_news_csv/'}
            n_class_dict = {'yahoo': 10, 'yelp': 2, 'amazon': 2, 'dbpedia': 14, 'agnews': 4}

            # Others

            print('dataset_name: ', dataset_name)
            data[t] = {}
            data[t]['name'] = dataset_name
            data[t]['ncla'] = n_class_dict[dataset_name]

            taskcla.append((t, int(data[t]['ncla'])))

            datasets = get_dataset_sup(path_dict[dataset_name] + 'train.csv', n_class_dict[dataset_name], dataset_name,
                                       sample_num_per_class=args.sample_num_per_class, need_split=True)
            train_datasets = datasets['train']
            dev_datasets = datasets['test']
            datasets = get_dataset_sup(path_dict[dataset_name] + 'test.csv', n_class_dict[dataset_name], dataset_name,
                                       sample_num_per_class=100)
            test_datasets = datasets['train']

            new_column = [t] * len(train_datasets)
            train_datasets = train_datasets.add_column("task", new_column)

            new_column = [t] * len(dev_datasets)
            dev_datasets = dev_datasets.add_column("task", new_column)

            new_column = [t] * len(test_datasets)
            test_datasets = test_datasets.add_column("task", new_column)

            raw_datasets = DatasetDict({
                'train': train_datasets,
                'validation':dev_datasets,
                'test': test_datasets,
            })

        elif dataset_name in args.dialogue_datasets:
            datasets = {}
            for ds in ['train', 'test', 'validation']:
                datasets[ds] = {}
                datasets[ds]['source'] = []
                datasets[ds]['target'] = []
                datasets[ds]['task'] = []


                json_path = os.path.join(args.base_dir + '/data_dialogue/'+dataset_name,ds+'.jsonl')
                with jsonlines.open(json_path,'r') as f_reader:
                    source_col_name = "history"
                    target_col_name = "reply"

                    instance_num = 0
                    for f in f_reader:
                        datasets[ds]['source'].append(f[source_col_name])
                        datasets[ds]['target'].append(f[target_col_name])
                        datasets[ds]['task'].append(t)
                        instance_num+= 1

                        # if args.sample_cap is not None and instance_num > args.sample_cap:
                        #     break # I don't want to have too many instances

            print('datasets["train"]: ',len(datasets["train"]['task']))
            print('datasets["test"]: ',len(datasets["test"]['task']))

            raw_datasets = DatasetDict({
                'train': Dataset.from_dict(datasets["train"]),
                'validation': Dataset.from_dict(datasets["validation"]),
                'test': Dataset.from_dict(datasets["test"]),
            })
            taskcla.append((t, 1))

        elif dataset_name in args.ner_datasets:
            datasets = {}
            for ds in ['train', 'test', 'validation']:
                datasets[ds] = {}
                datasets[ds]['source'] = []
                datasets[ds]['target'] = []
                datasets[ds]['task'] = []
                datasets[ds]['cls_labels'] = []


                json_path = os.path.join(args.base_dir + '/data_ner/'+dataset_name,ds+'.jsonl')
                with jsonlines.open(json_path,'r') as f_reader:
                    source_col_name = "tokens"
                    target_col_name = "labels"

                    instance_num = 0
                    for f in f_reader:
                        datasets[ds]['source'].append(f[source_col_name])
                        datasets[ds]['target'].append(f[target_col_name])
                        datasets[ds]['cls_labels'].append(f[target_col_name])
                        datasets[ds]['task'].append(t)
                        instance_num+= 1

                        if args.sample_cap is not None and instance_num > args.sample_cap and ds=='train': # very unstable when cutting the test
                            break # I don't want to have too many instances

            taskcla.append((t, len(set(
                [y for x in datasets['train']['cls_labels'] for y in x]

            ))))

            print('datasets["train"]: ',len(datasets["train"]['task']))
            print('datasets["test"]: ',len(datasets["test"]['task']))

            raw_datasets = DatasetDict({
                'train': Dataset.from_dict(datasets["train"]),
                'validation': Dataset.from_dict(datasets["validation"]),
                'test': Dataset.from_dict(datasets["test"]),
            })


        else:
            datasets = {}
            for ds in ['train', 'test', 'validation']:
                if ds == 'validation':
                    ds_ = 'valid'
                else:
                    ds_ = ds


                datasets[ds] = {}
                datasets[ds]['source'] = []
                datasets[ds]['target'] = []
                datasets[ds]['task'] = []

                json_path = os.path.join(args.base_dir + '/data_sum/'+args.dataset_type+'/'+dataset_name,ds_+'.jsonl')
                with jsonlines.open(json_path,'r') as f_reader:
                    if dataset_name == 'qmsum':
                        target_col_name = 'tgt'
                        source_col_name = 'src'
                    else:
                        target_col_name = 'summary'
                        source_col_name = 'document'

                    instance_num = 0
                    for f in f_reader:
                        datasets[ds]['source'].append(f[source_col_name])
                        datasets[ds]['target'].append(f[target_col_name])
                        datasets[ds]['task'].append(t)
                        instance_num+= 1
                        # if dataset_name == 'qmsum' and instance_num > 200:
                        #     break # I don't want to have too many instances
                        # if args.sample_cap is not None and instance_num > args.sample_cap:
                        #     break # I don't want to have too many instances


            print('datasets["train"]: ',len(datasets["train"]['task']))

            raw_datasets = DatasetDict({
                'train': Dataset.from_dict(datasets["train"]),
                'validation': Dataset.from_dict(datasets["validation"]),
                'test': Dataset.from_dict(datasets["test"]),
            })

            taskcla.append((t, 1))

        for s in ['train', 'test', 'dev']:  # all test

            data[t][s] = {}

            if s == 'train':
                data[t][s] = raw_datasets['train']
            elif s == 'dev':
                data[t][s] = raw_datasets['validation']
            elif s == 'test':
                data[t][s] = raw_datasets['test']

    return data,taskcla




def get_dataset_sup(path,num_class, dataset, sample_num_per_class=None,need_split=False):

    datasets = load_dataset('csv', data_files=path)
    sampler_per_class = len(datasets['train']) // num_class

    if sample_num_per_class is not None:
        train_idx = []
        for i in range(0, num_class):
            cls_idx = []
            random.seed(2021)
            for sample in range(sample_num_per_class):
                idx = random.randint(0 + i * sampler_per_class, sampler_per_class + i * sampler_per_class - 1)
                if idx not in cls_idx:
                    cls_idx.append(idx)
                else:
                    sample -= 1
            train_idx += cls_idx.copy()

    if 'yahoo' in dataset:

        def combine_function(example):
            for i in range(1, 4):
                if example[str(i)] is None:
                    example[str(i)] = ''

            example['source'] = example['1'] + ' ' + example['2'] + ' ' + example['3']
            example['cls_labels'] = example['0'] - 1  # this dataset's label index starts from 1
            example['target'] = str(example['0'] ) # TODO: just placeholder, need to change if we want to use it

            return example

        datasets = datasets.map(combine_function,
                                batched=False,
                                num_proc=16,
                                remove_columns=['0', '1', '2', '3'],
                                )

    elif 'yelp' in dataset:
        def combine_function(example):
            for i in range(1, 2):
                if example[str(i)] is None:
                    example[str(i)] = ''

            example['source'] = example['1']
            example['cls_labels'] = example['0'] - 1  # this dataset's label index starts from 1
            example['target'] = str(example['0'] ) # TODO: just placeholder, need to change if we want to use it

            return example

        datasets = datasets.map(combine_function,
                                batched=False,
                                num_proc=16,
                                remove_columns=['0', '1'],
                                )

    elif 'dbpedia' in dataset or 'agnews' in dataset or 'amazon' in dataset:
        def combine_function(example):
            for i in range(1, 3):
                if example[str(i)] is None:
                    example[str(i)] = ''

            example['source'] = example['1'] + ' ' + example['2']
            example['cls_labels'] = example['0'] - 1  # this dataset's label index starts from 1
            example['target'] = str(example['0'] ) # TODO: just placeholder, need to change if we want to use it

            return example

        datasets = datasets.map(combine_function,
                                batched=False,
                                num_proc=16,
                                remove_columns=['0', '1', '2'],
                                )

    if sample_num_per_class is not None:
        datasets['train'] = datasets['train'].select(train_idx)

    if need_split:
        datasets = datasets['train'].train_test_split(test_size=0.1, seed=2021, shuffle=True)

    print('datasets: ',datasets)

    return datasets

