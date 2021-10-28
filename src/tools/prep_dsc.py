import nltk
import numpy as np
import json
from collections import defaultdict
import xml.etree.ElementTree as ET
import random
import os
random.seed(1337)
np.random.seed(1337)

"""TODO: this file is not well-tested but just copied from another repository.
"""


polar_idx={'-1': 0, '1': 1,}
idx_polar={0: '-1', 1: '1'}


def parse_nli(fn,train_data_size=None):
    id=0
    corpus = []
    with open(fn) as senetence_file:
        sentences = senetence_file.readlines()

        # if train_data_size is not None: #TODO: if you want cut
        #     random.Random(0).shuffle(sentences) #more robust
        #     sentences = sentences[:train_data_size]

        for sentence in sentences:
            sentence1 = sentence.split('\t')[0]
            label = sentence.split('\t')[1].replace('\n','')
            # print('label: ',label)
            #
            if label not in polar_idx: continue
            corpus.append({"id": id, "sentence": sentence1, "term":None, "polarity": label})
            id+=1
    return corpus


domains = ['Video_Games','Toys_and_Games','Tools_and_Home_Improvement','Sports_and_Outdoors','Pet_Supplies',
           'Patio_Lawn_and_Garden','Office_Products','Musical_Instruments','Movies_and_TV',
           'Kindle_Store','Home_and_Kitchen','Health_and_Personal_Care','Grocery_and_Gourmet_Food','Electronics',
           'Digital_Music','Clothing_Shoes_and_Jewelry','Cell_Phones_and_Accessories','CDs_and_Vinyl',
           'Books','Beauty','Baby','Automotive','Apps_for_Android','Amazon_Instant_Video']

for domain in domains:
    if not os.path.isdir('./dat/dsc/'+domain):
        os.makedirs('./dat/dsc/'+domain)

    train_corpus=parse_nli('./data/dsc/'+domain+'_train.tsv')
    print('train_corpus: ',len(train_corpus))
    with open("./dat/dsc/"+domain+"/train.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus}, fw, sort_keys=True, indent=4)

    dev_corpus=parse_nli('./data/dsc/'+domain+'_dev.tsv')
    print('dev_corpus: ',len(dev_corpus))
    with open("./dat/dsc/"+domain+"/dev.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in dev_corpus}, fw, sort_keys=True, indent=4)

    test_corpus=parse_nli('./data/dsc/'+domain+'_test.tsv')
    print('test_corpus: ',len(test_corpus))
    with open("./dat/dsc/"+domain+"/test.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)
