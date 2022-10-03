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
import json


location = '/sdb/zke4'
data = ['ieer', 'btc', 'gum', 'ritter', 're3d', 'wnut2017', 'wikigold', 'conll2003', 'ontonote']
split = ['train','test','dev']

for d in data:
    for s in split:
        if s == 'dev':
            s_ = 'validation'
        else:
            s_ = s
        with jsonlines.open(os.path.join(location+ '/data_ner/'+d,s_+'.jsonl'),'w') as f_json:
            print(os.path.join(location+ '/data_ner/'+d,s_+'.jsonl'))
            with open(os.path.join(location + '/data_ner/'+d,s+'.json')) as f:
                json_file = json.load(f)
                for instance_key,instance_value in json_file.items():
                    f_json.write({'tokens':instance_value['tokens'],'labels':instance_value['labels']})
