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



#TODO: load sentence sentiment classification


train_rate = 0.8
valid_rate = 0.1
test_rate = 0.1
polar_idx={'0': 0, '1': 1, '2': 2}
idx_polar={0: '0', 1: '1', 2: '2'}


def parse_nli(fn):
    id=0
    corpus = []
    with open(fn) as senetence_file:
        sentences = senetence_file.readlines()
        for sentence in sentences:
            sentence1 = sentence[1:].replace('\n','')
            label = sentence[0].replace('\n','')
            # print('label: ',label)
            #
            if label not in polar_idx: continue
            corpus.append({"id": id, "sentence": sentence1, "term":None, "polarity": label})
            id+=1
    return corpus


domains = ['airconditioner','bike','diaper','GPS','headphone',
           'hotel','luggage','smartphone','stove','TV']

for domain in domains:
    if not os.path.isdir('./dat/ssc/'+domain):
        os.makedirs('./dat/ssc/'+domain)

    train_corpus=parse_nli('./data/ssc/'+domain)

    with open("./dat/ssc/"+domain+"/train.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)] }, fw, sort_keys=True, indent=4)
    with open("./dat/ssc/"+domain+"/dev.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):int(len(train_corpus)*(train_rate+valid_rate))] }, fw, sort_keys=True, indent=4)
    with open("./dat/ssc/"+domain+"/test.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*(train_rate+valid_rate)):] }, fw, sort_keys=True, indent=4)
