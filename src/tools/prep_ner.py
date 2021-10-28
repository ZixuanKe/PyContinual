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

# valid_split=150



def change_beginning_to_B(label):
    changed = False # some dataset does not have beginning B
    for lab_id,lab in enumerate(label):
        if lab!='O' and 'B' not in lab and changed == False:
            label[lab_id] = label[lab_id].replace('I-','B-')
            changed = True
        elif lab == 'O':
            changed = False
    return label

def parse_ner(filename):
    '''
    read file
    '''
    f = open(filename)
    sentence = []
    label= []
    id=0
    corpus = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n" or len(line.split())==0:
            if len(sentence) > 0:
                if 'wikigold' in filename: #only for wikigold, which does not have B
                    label = change_beginning_to_B(label)

                print('sentence: ',sentence)
                print('label: ',label)

                corpus.append({"id": id, "tokens": sentence, "labels": label})
                sentence = []
                label = []
                id+=1
            continue
        splits = line.split() #split for get the label, we are differnet
        if len(splits) < 2: continue
        sentence.append(splits[0])
        label.append(splits[-1])

    if len(sentence) >0:
        # data.append((sentence,label))
        corpus.append({"id": id, "tokens": sentence, "labels": label})
        id+=1
        print('sentence: ',sentence)
        print('label: ',label)

    return corpus





# wnut 2017

print('================= wnut2017 =========================')

fn_read = './data/ner/wnut2017/'
fn_write = './dat/ner/wnut2017/'
files_read = ['emerging.test.annotated','wnut17train.conll','emerging.dev.conll']

train_corpus = parse_ner(fn_read+'wnut17train.conll')
with open(fn_write+"/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus}, fw, sort_keys=True, indent=4)

dev_corpus = parse_ner(fn_read+'emerging.dev.conll')
with open(fn_write+"/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in dev_corpus }, fw, sort_keys=True, indent=4)

test_corpus = parse_ner(fn_read+'emerging.test.annotated')
with open(fn_write+"/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus }, fw, sort_keys=True, indent=4)


# wikigold

print('================= wikigold =========================')

fn_read = './data/ner/wikigold/'
fn_write = './dat/ner/wikigold/'

train_rate = 0.8
valid_rate = 0.1
test_rate = 0.1

train_corpus = parse_ner(fn_read+'wikigold.conll.txt')
with open(fn_write+"/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)]}, fw, sort_keys=True, indent=4)
with open(fn_write+"/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):int(len(train_corpus)*(train_rate+valid_rate))]}, fw, sort_keys=True, indent=4)
with open(fn_write+"/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*(train_rate+valid_rate)):]}, fw, sort_keys=True, indent=4)



#re3d

print('================= re3d =========================')


fn_read = './data/ner/re3d/'
fn_write = './dat/ner/re3d/'

train_rate = 0.9
valid_rate = 0.1

train_corpus = parse_ner(fn_read+'re3d-train.conll')
with open(fn_write+"/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)]}, fw, sort_keys=True, indent=4)
with open(fn_write+"/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):int(len(train_corpus)*(train_rate+valid_rate))]}, fw, sort_keys=True, indent=4)

test_corpus = parse_ner(fn_read+'re3d-test.conll')
with open(fn_write+"/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus }, fw, sort_keys=True, indent=4)


# Ritter twitter

print('================= ritter =========================')


fn_read = './data/ner/ritter/'
fn_write = './dat/ner/ritter/'

train_rate = 0.8
valid_rate = 0.1
test_rate = 0.1

train_corpus = parse_ner(fn_read+'ner.txt')
with open(fn_write+"/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)]}, fw, sort_keys=True, indent=4)
with open(fn_write+"/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):int(len(train_corpus)*(train_rate+valid_rate))]}, fw, sort_keys=True, indent=4)
with open(fn_write+"/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*(train_rate+valid_rate)):]}, fw, sort_keys=True, indent=4)


#gum

print('================= gum =========================')


fn_read = './data/ner/gum/'
fn_write = './dat/ner/gum/'

train_rate = 0.9
valid_rate = 0.1

train_corpus = parse_ner(fn_read+'gum-train.conll')
with open(fn_write+"/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)]}, fw, sort_keys=True, indent=4)
with open(fn_write+"/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):int(len(train_corpus)*(train_rate+valid_rate))]}, fw, sort_keys=True, indent=4)

test_corpus = parse_ner(fn_read+'gum-test.conll')
with open(fn_write+"/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus }, fw, sort_keys=True, indent=4)


#btc

print('================= btc =========================')


fn_read = './data/ner/btc/'
fn_write = './dat/ner/btc/'

train_rate = 0.8
valid_rate = 0.1
test_rate = 0.1

train_corpus = parse_ner(fn_read+'btc.txt')
random.Random(0).shuffle(train_corpus) #more robust

with open(fn_write+"/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)]}, fw, sort_keys=True, indent=4)
with open(fn_write+"/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):int(len(train_corpus)*(train_rate+valid_rate))]}, fw, sort_keys=True, indent=4)
with open(fn_write+"/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*(train_rate+valid_rate)):]}, fw, sort_keys=True, indent=4)


#ieer

print('================= ieer =========================')


fn_read = './data/ner/ieer/'
fn_write = './dat/ner/ieer/'


train_rate = 0.8
valid_rate = 0.1
test_rate = 0.1

train_corpus = parse_ner(fn_read+'ieer.txt')

train_corpus = parse_ner(fn_read+'ieer.txt')
random.Random(0).shuffle(train_corpus) #more robust


with open(fn_write+"/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[:int(len(train_corpus)*train_rate)]}, fw, sort_keys=True, indent=4)
with open(fn_write+"/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*train_rate):int(len(train_corpus)*(train_rate+valid_rate))]}, fw, sort_keys=True, indent=4)
with open(fn_write+"/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus[int(len(train_corpus)*(train_rate+valid_rate)):]}, fw, sort_keys=True, indent=4)



#ontonote: nothing need to do

print('================= ontonote =========================')

fn_read = './data/ner/ontonote/'
fn_write = './dat/ner/ontonote/'

train_corpus = parse_ner(fn_read+'onto.train.ner')
with open(fn_write+"/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus}, fw, sort_keys=True, indent=4)

dev_corpus = parse_ner(fn_read+'onto.development.ner')
with open(fn_write+"/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in dev_corpus }, fw, sort_keys=True, indent=4)

test_corpus = parse_ner(fn_read+'onto.test.ner')
with open(fn_write+"/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus }, fw, sort_keys=True, indent=4)


#conll2003: nothing need to do

print('================= conll2003 =========================')


fn_read = './data/ner/conll2003/'
fn_write = './dat/ner/conll2003/'

train_corpus = parse_ner(fn_read+'train.txt')
with open(fn_write+"/train.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in train_corpus}, fw, sort_keys=True, indent=4)

dev_corpus = parse_ner(fn_read+'valid.txt')
with open(fn_write+"/dev.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in dev_corpus }, fw, sort_keys=True, indent=4)

test_corpus = parse_ner(fn_read+'test.txt')
with open(fn_write+"/test.json", "w") as fw:
    json.dump({rec["id"]: rec for rec in test_corpus }, fw, sort_keys=True, indent=4)
