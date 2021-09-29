#Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import BertTokenizer as BertTokenizer
import os
import torch
import numpy as np
import gzip
import nlp_data_utils as data_utils
from os import path
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from w2v_util import Tokenizer
import json
import math
import random
from datasets import load_dataset

from nlp_data_utils import ABSATokenizer

classes_20newsgroup = [
    "19997_comp.graphics",
    "19997_comp.os.ms-windows.misc",
    "19997_comp.sys.ibm.pc.hardware",
    "19997_comp.sys.mac.hardware",
    "19997_comp.windows.x",
    "19997_rec.autos",
    "19997_rec.motorcycles",
    "19997_rec.sport.baseball",
    "19997_rec.sport.hockey",
    "19997_sci.crypt",
    "19997_sci.electronics",
    "19997_sci.med",
    "19997_sci.space",
    "19997_misc.forsale",
    "19997_talk.politics.misc",
    "19997_talk.politics.guns",
    "19997_talk.politics.mideast",
    "19997_talk.religion.misc",
    "19997_alt.atheism",
    "19997_soc.religion.christian",
]

dataset_name = 'newsgroup'

#TODO: need change for w2v version (embedding and feature generation)



def embedding_generation(args):

    newsgroup_str = []
    for c_id,cla in enumerate(classes_20newsgroup):
        d = load_dataset(dataset_name,cla, split='train')
        d_split = d.train_test_split(test_size=0.2,shuffle=True,seed=args.seed)
        newsgroup_str += d_split['train']['text']

        d_split = d_split['test'].train_test_split(test_size=0.5,shuffle=True,seed=args.seed) # test into half-half

        newsgroup_str += d_split['test']['text']
        newsgroup_str += d_split['train']['text']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(newsgroup_str)
    tokenizer.texts_to_sequences(newsgroup_str)
    word_index = tokenizer.word_index

    # print('word_index: ',word_index)
    print('Found %s unique tokens.' % len(word_index))

    if not path.exists("./dat/newsgroup/w2v_embedding"):
        embeddings_index = {}
        # f = gzip.open('./cc.en.300.vec.gz')
        f = open('./amazon_review_300d.vec','r')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            # embeddings_index[word.decode("utf-8")] = coefs
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # exit()

        embeddings = torch.from_numpy(embedding_matrix)
        vocab_size = len(word_index)

        torch.save(embeddings,'./dat/newsgroup/w2v_embedding')
        torch.save(vocab_size,'./dat/newsgroup/vocab_size')
    else:
        embeddings = torch.load('./dat/newsgroup/w2v_embedding')
        vocab_size = torch.load('./dat/newsgroup/vocab_size')

    return embeddings,vocab_size,tokenizer

def get(logger=None,args=None):

    embeddings,vocab_size,tokenizer = embedding_generation(args)
    dataset_name = 'newsgroup'
    classes = classes_20newsgroup

    print('dataset_name: ',dataset_name)

    data={}
    taskcla=[]

    # Others
    f_name = dataset_name + '_random_'+str(args.ntasks)
    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    print('random_sep: ',random_sep)

    dataset = {}
    dataset['train'] = {}
    dataset['valid'] = {}
    dataset['test'] = {}



    for c_id,cla in enumerate(classes):
        d = load_dataset(dataset_name,cla, split='train')
        d_split = d.train_test_split(test_size=0.2,shuffle=True,seed=args.seed)
        dataset['train'][c_id] = d_split['train']

        d_split = d_split['test'].train_test_split(test_size=0.5,shuffle=True,seed=args.seed) # test into half-half

        dataset['test'][c_id] = d_split['test']
        dataset['valid'][c_id] = d_split['train']

    class_per_task = args.class_per_task

    # print('len(dataset): ',len(dataset))
    # print('dataset: ',len(dataset['train'][0]))
    # print('dataset: ',len(dataset['valid'][0]))
    # print('dataset: ',len(dataset['test'][0]))

    #TODO: seq_length (current 128)
    #TODO: do we need any data cleaning


    examples = {}
    for s in ['train','test','valid']:
        examples[s] = {}
        for c_id, c_data in dataset[s].items():
            nn=(c_id//class_per_task) #which task_id this class belongs to

            if nn not in examples[s]: examples[s][nn] = []
            for c_dat in c_data:
                text= c_dat['text']
                label = c_id%class_per_task
                examples[s][nn].append((text,label))

    for t in range(args.ntasks):
        t_seq = int(random_sep[t].split('_')[-1])
        data[t]={}
        data[t]['ncla']=class_per_task
        data[t]['name']=dataset_name+'_'+str(t_seq)
        taskcla.append((t,int(data[t]['ncla'])))

        for s in ['train','test','valid']:
            if s == 'train':
                processor = data_utils.DtcProcessor()
                label_list = processor.get_labels(args.ntasks)
                train_examples =  processor._create_examples(examples[s][t_seq], "train")

                #TODO: in case you want to cut data, insert here

                num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs
                # num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

                train_features = data_utils.convert_examples_to_features_w2v_dsc(
                    train_examples, label_list,tokenizer,args)
                logger.info("***** Running training *****")
                logger.info("  Num examples = %d", len(train_examples))
                logger.info("  Batch size = %d", args.train_batch_size)
                logger.info("  Num steps = %d", num_train_steps)

                all_tokens_term_ids = torch.tensor([f.tokens_term_ids for f in train_features], dtype=torch.long)
                all_tokens_sentence_ids = torch.tensor([f.tokens_sentence_ids for f in train_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
                train_data = TensorDataset(
                    all_tokens_term_ids,
                    all_tokens_sentence_ids,
                    all_label_ids)


                data[t]['train'] = train_data
                data[t]['num_train_steps']=num_train_steps


            if s == 'valid':

                valid_examples = processor._create_examples(examples[s][t_seq], "valid")
                valid_features=data_utils.convert_examples_to_features_w2v_dsc(
                    valid_examples, label_list, tokenizer,args)

                valid_all_tokens_term_ids = torch.tensor([f.tokens_term_ids for f in valid_features], dtype=torch.long)
                valid_all_tokens_sentence_ids = torch.tensor([f.tokens_sentence_ids for f in valid_features], dtype=torch.long)
                valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)

                valid_data = TensorDataset(
                    valid_all_tokens_term_ids,
                    valid_all_tokens_sentence_ids,
                    valid_all_label_ids)


                logger.info("***** Running validations *****")
                logger.info("  Num orig examples = %d", len(valid_examples))
                logger.info("  Num split examples = %d", len(valid_features))
                logger.info("  Batch size = %d", args.train_batch_size)

                data[t]['valid']=valid_data

            if s == 'test':
                processor = data_utils.DtcProcessor()
                label_list = processor.get_labels(args.ntasks)
                eval_examples = processor._create_examples(examples[s][t_seq], "test")
                eval_features = data_utils.convert_examples_to_features_w2v_dsc(eval_examples, label_list, tokenizer,args)

                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_tokens_term_ids = torch.tensor([f.tokens_term_ids for f in eval_features], dtype=torch.long) #this term is actually also the sentence
                all_tokens_sentence_ids = torch.tensor([f.tokens_sentence_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(
                    all_tokens_term_ids,
                    all_tokens_sentence_ids,
                    all_label_ids)

                # Run prediction for full data

                data[t]['test']=eval_data


    # total number of class
    n=0
    for t in data.keys():
        n+=data[t]['ncla']
    data['ncla']=n


    return data,taskcla,vocab_size,embeddings




