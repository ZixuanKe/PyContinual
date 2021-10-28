import os, sys
import numpy as np
import torch
from sklearn.utils import shuffle
import json
import xml.etree.ElementTree as ET

import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torchvision import datasets,transforms
import json
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image
import random
def get(seed=0):

    ninteen_domains = \
    [
        'Bing3domains_Speaker',
        'Bing3domains_Router',
        'Bing3domains_Computer',

        'Bing5domains_Nokia6610',
        'Bing5domains_NikonCoolpix4300',
        'Bing5domains_CreativeLabsNomadJukeboxZenXtra40GB',
        'Bing5domains_CanonG3',
        'Bing5domains_ApexAD2600Progressive',

        'Bing9domains_CanonPowerShotSD500',
        'Bing9domains_CanonS100',
        'Bing9domains_DiaperChamp',
        'Bing9domains_HitachiRouter',
        'Bing9domains_ipod',
        'Bing9domains_LinksysRouter',
        'Bing9domains_MicroMP3',
        'Bing9domains_Nokia6600',
        'Bing9domains_Norton',

        'XuSemEval14_rest',
        'XuSemEval14_laptop',

    ]



    ten_domains = \
    [
        'Bing3domains_Speaker',
        'Bing3domains_Router',

        'Bing5domains_Nokia6610',
        'Bing5domains_NikonCoolpix4300',
        'Bing5domains_CreativeLabsNomadJukeboxZenXtra40GB',

        'Bing9domains_CanonPowerShotSD500',
        'Bing9domains_CanonS100',
        'Bing9domains_DiaperChamp',

        'XuSemEval14_rest',
        'XuSemEval14_laptop',

    ]

    # ner_domains_oldseq = ['ieer','btc','gum','ritter','re3d', 'wnut2017','wikigold','conll2003']
    # ner_domains_8 = ['ieer','btc','gum','ritter','ontonote', 'wnut2017','wikigold','conll2003']
    # ner_domains_7 = ['ieer','btc','ritter','ontonote', 'wnut2017','wikigold','conll2003']
    # ner_domains_6 = ['ieer','btc','ontonote', 'wnut2017','wikigold','conll2003']
    ner_domains_full = ['ieer','btc','gum','ritter','re3d', 'wnut2017','wikigold','conll2003','ontonote']
    ner_domains_overlap = ['ieer','btc','gum','re3d','wikigold','conll2003','ontonote']

    # nli_domains = ['travel','telephone','slate','government','fiction']

    # dsc_domains = ['Video_Games','Toys_and_Games','Tools_and_Home_Improvement','Sports_and_Outdoors','Pet_Supplies',
    #        'Patio_Lawn_and_Garden','Office_Products','Musical_Instruments','Movies_and_TV',
    #        'Kindle_Store','Home_and_Kitchen','Health_and_Personal_Care','Grocery_and_Gourmet_Food','Electronics',
    #        'Digital_Music','Clothing_Shoes_and_Jewelry','Cell_Phones_and_Accessories','CDs_and_Vinyl',
    #        'Books','Beauty','Baby','Automotive','Apps_for_Android','Amazon_Instant_Video']

    # dsc_domains_10 = ['Video_Games','Toys_and_Games','Tools_and_Home_Improvement','Sports_and_Outdoors','Pet_Supplies',
    #            'Patio_Lawn_and_Garden','Office_Products','Musical_Instruments','Movies_and_TV',
    #            'Kindle_Store']

    # ssc_domains = ['airconditioner','bike','diaper','GPS','headphone',
               # 'hotel','luggage','smartphone','stove','TV']


    # with open('asc_random_19','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(ninteen_domains)
    #         f_random_seq.writelines('\t'.join(ninteen_domains) + '\n')

    # with open('asc_random_10','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(ten_domains)
    #         f_random_seq.writelines('\t'.join(ten_domains) + '\n')

    # with open('ner_random','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(ner_domains_6)
    #         f_random_seq.writelines('\t'.join(ner_domains_6) + '\n')

    # with open('ner_random','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(ner_domains_full)
    #         f_random_seq.writelines('\t'.join(ner_domains_full) + '\n')


    # with open('ner_random_overlap','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(ner_domains_overlap)
    #         f_random_seq.writelines('\t'.join(ner_domains_overlap) + '\n')

    # with open('nli_random','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(nli_domains)
    #         f_random_seq.writelines('\t'.join(nli_domains) + '\n')

    # with open('dsc_random_10','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(dsc_domains_10)
    #         f_random_seq.writelines('\t'.join(dsc_domains_10) + '\n')

    # with open('ssc_random','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(ssc_domains)
    #         f_random_seq.writelines('\t'.join(ssc_domains) + '\n')


    # newsgroup_domains_10 = ['newsgroup_'+str(x) for x in range(10)]
    # with open('newsgroup_random_10','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(newsgroup_domains_10)
    #         f_random_seq.writelines('\t'.join(newsgroup_domains_10) + '\n')

    #
    # celeba_10 = ['celeba-'+str(x) for x in range(10)]
    #
    # with open('celeba_10','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(celeba_10)
    #         f_random_seq.writelines('\t'.join(celeba_10) + '\n')

    # femnist_10 = ['femnist-'+str(x) for x in range(10)]
    #
    # with open('femnist_10','w') as f_random_seq:
    #     for repeat_num in range(20):
    #         random.shuffle(femnist_10)
    #         f_random_seq.writelines('\t'.join(femnist_10) + '\n')

    femnist_20 = ['femnist-'+str(x) for x in range(20)]

    with open('femnist_20','w') as f_random_seq:
        for repeat_num in range(20):
            random.shuffle(femnist_20)
            f_random_seq.writelines('\t'.join(femnist_20) + '\n')


if __name__ == "__main__":
    get()