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
import os





machine = '/sdb/zke4/seq0/seed2021/one/ccdv/cnn_dailymail/ccdv'
# metric = 'rouge1'
# metric = 'rouge2'
# metric = 'rougeLsum'
dataset = 'dev'
seed = '2021'

for metric in ['rouge1','rouge2','rougeLsum']:
    with open(dataset+'_'+metric+'_summary_'+seed, 'w') as fw_file:
        for step in range(500,12500,500):
            with open(os.path.join(machine,dataset+'_progressive_'+metric+'_'+str(step)+'_'+seed),'r') as fr_file:

                fw_file.writelines(fr_file.readlines()[0])
