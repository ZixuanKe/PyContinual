import json
import os.path
import random

import jsonlines
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer



location = '/sdb/zke4/data_sum/full/'
data = ['icsi/','ami/']
split = ['train','test','val']

for d in data:
    for s in split:
        if s == 'val':
            s_ = 'valid'
        else:
            s_ = s
        with jsonlines.open(location+d+s_+'.jsonl','w') as f_json:
            with open(location+d+s+'.source', 'r') as f_source,open(location+d+s+'.target', 'r') as f_target:
                sources = f_source.readlines()
                targets = f_target.readlines()

                print('sources: ',len(sources))
                print('targets: ',len(targets))

                for id, source in enumerate(sources):
                    f_json.write({'document':source,'summary':targets[id]})


