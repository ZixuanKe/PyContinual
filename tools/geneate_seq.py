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



new_data = {}
# data = ['ieer', 'btc', 'gum', 'ritter', 're3d', 'wnut2017', 'wikigold', 'conll2003', 'ontonote']
# data = ['sgd_services','sgd_flights','sgd_buses','sgd_ridesharing','sgd_rentalcars',
#         'sgd_homes','sgd_music','sgd_events','sgd_banks','sgd_hotels','sgd_calendar',
#         'sgd_media','sgd_movies','sgd_restaurants','sgd_alarm','sgd_weather',
#         'sgd_travel','sgd_payment','sgd_trains']

# data = ['tma_movie','tma_auto', 'tma_restaurant', 'tma_pizza', 'tma_uber', 'tma_coffee',
#         'tmb_hotel','tmb_movie','tmb_flight','tmb_sport',
#         'tmb_restaurant','tmb_music','tmb_food-ordering']


# data = ['conll2003','wikigold','btc','re3d','gum']

data = ['nyt','stack','emails', 'reddit','icsi'  , 'ami'  ]
#
# data = ['conll2003','wikigold','btc','re3d','gum',
#
#         'MWOZ_taxi','MWOZ_hotel','MWOZ_attraction','MWOZ_train','MWOZ_restaurant',
#
#         'XuSemEval14_rest',
#          'XuSemEval14_laptop',
#
#          'Bing3domains_Speaker',
#          'Bing3domains_Router',
#          'Bing3domains_Computer',
#
#          'Bing5domains_Nokia6610',
#          'Bing5domains_NikonCoolpix4300',
#          'Bing5domains_CreativeLabsNomadJukeboxZenXtra40GB',
#          'Bing5domains_CanonG3',
#          'Bing5domains_ApexAD2600Progressive',
#
#          'Bing9domains_CanonPowerShotSD500',
#          'Bing9domains_CanonS100',
#          'Bing9domains_DiaperChamp',
#          'Bing9domains_HitachiRouter',
#          'Bing9domains_ipod',
#          'Bing9domains_LinksysRouter',
#          'Bing9domains_MicroMP3',
#          'Bing9domains_Nokia6600',
#          'Bing9domains_Norton',
#
#         'nyt',
#         'stack',
#         'emails',
#         'reddit',
#         'icsi',
#         'ami',
#
#         'yelp',
#         'amazon',
#         'yahoo',
#         'dbpedia',
#         'agnews'
#
#         ]
# print('length: ', len(data))

with open('summarization_conv_trans', 'w') as f_random_seq:
    for repeat_num in range(10):
        random.shuffle(data)
        f_random_seq.writelines('\t'.join(data) + '\n')



