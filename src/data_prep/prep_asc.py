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

valid_split=150
train_rate = 0.8
valid_rate = 0.1
test_rate = 0.1

polar_idx={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}
idx_polar={0: 'positive', 1: 'negative', 2: 'neutral'}


def parse_Bing(fn):
    id=0
    corpus = []
    with open(fn,'r') as review_file:
        reviews = review_file.readlines()
        for review in reviews:
            opins=set()
            if review[:2] != '##' and '##' in review and '[' in review and \
                    ('+' in review.split('##')[0] or '-' in review.split('##')[0]):
                # print(review.split('##')[0])
                current_sentence = review.split('##')[1][:-1].replace('\t',' ')
                #aspects: may be more than one
                aspect_str = review.split('##')[0]
                if ',' in aspect_str:
                    aspect_all = aspect_str.split(',')
                    for each_aspect in aspect_all:
                        current_aspect = each_aspect.split('[')[0]
                        current_sentiment = each_aspect.split('[')[1][0]
                        opins.add((current_aspect, current_sentiment))

                elif ',' not in aspect_str:
                    current_aspect = aspect_str.split('[')[0]
                    current_sentiment = aspect_str.split('[')[1][0]
                    opins.add((current_aspect, current_sentiment))

                for ix, opin in enumerate(opins):
                    corpus.append({"id": id, "sentence": current_sentence, "term": opin[0], "polarity": opin[-1]})
                    id+=1
    return corpus


domains = ['CanonPowerShotSD500','CanonS100','DiaperChamp','HitachiRouter','ipod', \
           'LinksysRouter','MicroMP3','Nokia6600','Norton']
for domain in domains:
    train_corpus=parse_Bing('./data/Review9Domains/'+domain+'.txt')
    print('train_corpus: ',len(train_corpus))
    with open("./dat/Bing9Domains/asc/"+domain+"/train.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)] }, fw, sort_keys=True, indent=4)
    with open("./dat/Bing9Domains/asc/"+domain+"/dev.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):int(len(train_corpus)*(train_rate+valid_rate))] }, fw, sort_keys=True, indent=4)
    with open("./dat/Bing9Domains/asc/"+domain+"/test.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*(train_rate+valid_rate)):] }, fw, sort_keys=True, indent=4)


domains = ['Nokia6610','NikonCoolpix4300','CreativeLabsNomadJukeboxZenXtra40GB','CanonG3','ApexAD2600Progressive']
for domain in domains:
    train_corpus=parse_Bing('./data/Review5Domains/'+domain+'.txt')
    with open("./dat/Bing5Domains/asc/"+domain+"/train.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)] }, fw, sort_keys=True, indent=4)
    with open("./dat/Bing5Domains/asc/"+domain+"/dev.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):int(len(train_corpus)*(train_rate+valid_rate))] }, fw, sort_keys=True, indent=4)
    with open("./dat/Bing5Domains/asc/"+domain+"/test.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*(train_rate+valid_rate)):] }, fw, sort_keys=True, indent=4)

domains = ['Speaker','Router','Computer']
for domain in domains:
    train_corpus=parse_Bing('./data/Review3Domains/'+domain+'.txt')
    with open("./dat/Bing3Domains/asc/"+domain+"/train.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)] }, fw, sort_keys=True, indent=4)
    with open("./dat/Bing3Domains/asc/"+domain+"/dev.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):int(len(train_corpus)*(train_rate+valid_rate))] }, fw, sort_keys=True, indent=4)
    with open("./dat/Bing3Domains/asc/"+domain+"/test.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*(train_rate+valid_rate)):] }, fw, sort_keys=True, indent=4)
