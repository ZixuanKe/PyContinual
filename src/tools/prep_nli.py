import nltk
import numpy as np
import json
from collections import defaultdict
import xml.etree.ElementTree as ET
import random
random.seed(1337)
np.random.seed(1337)

"""TODO: this file is not well-tested but just copied from another repository.
"""

train_rate = 0.9
valid_rate = 0.1

polar_idx={'neutral': 0, 'entailment': 1, 'contradiction': 2}
idx_polar={0: 'neutral', 1: 'entailment', 2: 'contradiction'}


def parse_nli(fn,domain):
    id=0
    corpus = []
    with open(fn) as senetence_file:
        sentences = senetence_file.readlines()[1:]
        for sentence in sentences:
            if sentence.split('\t')[9] != domain: continue
            sentence1 = sentence.split('\t')[5]
            sentence2 = sentence.split('\t')[6]
            label = sentence.split('\t')[0]

            # print('sentence1: ',sentence1)
            # print('sentence2: ',sentence2)
            # print('label: ',label)
            #
            if label not in polar_idx: continue
            corpus.append({"id": id, "sentence": sentence1, "term":sentence2, "polarity": label})
            id+=1
    return corpus


domains = ['slate','government','telephone','travel','fiction']
for domain in domains:
    train_corpus=parse_nli('./data/nli/multinli_1.0_train.txt',domain)
    print('train_corpus: ',len(train_corpus))
    with open("./dat/nli/"+domain+"/train.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)] }, fw, sort_keys=True, indent=4)
    with open("./dat/nli/"+domain+"/dev.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):] }, fw, sort_keys=True, indent=4)

for domain in domains:
   test_corpus=parse_nli('./data/nli/multinli_1.0_dev_matched.txt',domain)
   print('test_corpus: ',len(test_corpus))
   with open("./dat/nli/"+domain+"/test.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)

