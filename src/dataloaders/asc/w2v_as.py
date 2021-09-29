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


#"as" means aspect and sentence together into consideration for w2v


raw_datasets = [
            './data/absa/XuSemEval/14/rest',
            './data/absa/XuSemEval/14/laptop',
            './data/absa/Bing3Domains/',
            './data/absa/Bing5Domains/',
            './data/absa/Bing9Domains/',

            ]


datasets = [
            './dat/absa/XuSemEval/asc/14/rest',
            './dat/absa/XuSemEval/asc/14/laptop',

            './dat/absa/Bing3Domains/asc/Speaker',
            './dat/absa/Bing3Domains/asc/Router',
            './dat/absa/Bing3Domains/asc/Computer',

            './dat/absa/Bing5Domains/asc/Nokia6610',
            './dat/absa/Bing5Domains/asc/NikonCoolpix4300',
            './dat/absa/Bing5Domains/asc/CreativeLabsNomadJukeboxZenXtra40GB',
            './dat/absa/Bing5Domains/asc/CanonG3',
            './dat/absa/Bing5Domains/asc/ApexAD2600Progressive',

            './dat/absa/Bing9Domains/asc/CanonPowerShotSD500',
            './dat/absa/Bing9Domains/asc/CanonS100',
            './dat/absa/Bing9Domains/asc/DiaperChamp',
            './dat/absa/Bing9Domains/asc/HitachiRouter',
            './dat/absa/Bing9Domains/asc/ipod',
            './dat/absa/Bing9Domains/asc/LinksysRouter',
            './dat/absa/Bing9Domains/asc/MicroMP3',
            './dat/absa/Bing9Domains/asc/Nokia6600',
            './dat/absa/Bing9Domains/asc/Norton',
            ]



domains = [
     'XuSemEval14_rest',
     'XuSemEval14_laptop',

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
     'Bing9domains_Norton']


def embedding_generation(args):
    asc_str = []
    for dataset in raw_datasets:
        for file_name in os.listdir(dataset):
            with open(dataset+'/'+file_name) as asc_file:
                if 'json' in file_name:
                    lines = json.load(asc_file)
                    for (i, ids) in enumerate(lines):
                        asc_str.append(lines[ids]['sentence']+' ' +lines[ids]['term'])
                else:
                    reviews = asc_file.readlines()
                    for review in reviews:
                        opins=set()
                        if review[:2] != '##' and '##' in review and '[' in review and \
                                ('+' in review.split('##')[0] or '-' in review.split('##')[0]):
                            current_sentence = review.split('##')[1][:-1].replace('\t',' ')
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
                                asc_str.append(current_sentence + ' ' + opin[0])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(asc_str)
    tokenizer.texts_to_sequences(asc_str)
    word_index = tokenizer.word_index

    # print('word_index: ',word_index)
    print('Found %s unique tokens.' % len(word_index))

    if not path.exists("./dat/absa/w2v_as_embedding"):
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

        torch.save(embeddings,'./dat/absa/w2v_as_embedding')
        torch.save(vocab_size,'./dat/absa/w2v_as_vocab_size')
    else:
        embeddings = torch.load('./dat/absa/w2v_as_embedding')
        vocab_size = torch.load('./dat/absa/w2v_as_vocab_size')

    return embeddings,vocab_size,tokenizer

def get(logger=None,args=None):

    embeddings,vocab_size,tokenizer = embedding_generation(args)

    data={}
    taskcla=[]
    t=0

    for dataset in datasets:
        data[t]={}
        if 'Bing' in dataset:
            data[t]['name']=dataset
            if 'der' in args.approach or 'a-gem' in args.approach: data[t]['ncla']=3
            else: data[t]['ncla']=2
        elif 'XuSemEval' in dataset:
            data[t]['name']=dataset
            data[t]['ncla']=3




        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        train_examples = processor.get_train_examples(dataset)
        num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs

        train_features = data_utils.convert_examples_to_features_w2v_as(
            train_examples, label_list,tokenizer,args)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_tokens_term_ids = torch.tensor([f.tokens_term_ids for f in train_features], dtype=torch.long)
        all_tokens_sentence_ids = torch.tensor([f.tokens_sentence_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_tokens_term_sentence_ids = torch.tensor([f.tokens_term_sentence_ids for f in train_features], dtype=torch.long)

        # print('all_tokens_term_ids: ',all_tokens_term_ids)

        train_data = TensorDataset(
            all_tokens_term_ids,
            all_tokens_term_sentence_ids, # return term_sentence as sentence, thus no need to change other guys
            all_label_ids)


        data[t]['train'] = train_data
        data[t]['num_train_steps']=num_train_steps

        valid_examples = processor.get_dev_examples(dataset)
        valid_features=data_utils.convert_examples_to_features_w2v_as\
                (valid_examples, label_list, tokenizer,args)
        valid_all_tokens_term_ids = torch.tensor([f.tokens_term_ids for f in valid_features], dtype=torch.long)
        valid_all_tokens_sentence_ids = torch.tensor([f.tokens_sentence_ids for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_tokens_term_sentence_ids = torch.tensor([f.tokens_term_sentence_ids for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(
            valid_all_tokens_term_ids,
            valid_all_tokens_term_sentence_ids, # return term_sentence as sentence, thus no need to change other guys
            valid_all_label_ids)
        

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)


        data[t]['valid']=valid_data

        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        eval_examples = processor.get_test_examples(dataset)
        eval_features = \
            data_utils.convert_examples_to_features_w2v_as\
                (eval_examples, label_list, tokenizer,args)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        all_tokens_term_ids = torch.tensor([f.tokens_term_ids for f in eval_features], dtype=torch.long)
        all_tokens_sentence_ids = torch.tensor([f.tokens_sentence_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tokens_term_sentence_ids = torch.tensor([f.tokens_term_sentence_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(
            all_tokens_term_ids,
            all_tokens_term_sentence_ids, # return term_sentence as sentence, thus no need to change other guys
            all_label_ids)

        data[t]['test']=eval_data

        t+=1


    # Others
    f_name = 'asc_random'
    data_asc={}

    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    print('random_sep: ',random_sep)
    print('domains: ',domains)

    print('random_sep: ',len(random_sep))
    print('domains: ',len(domains))

    for task_id in range(args.ntasks):
        # print('task_id: ',task_id)
        asc_id = domains.index(random_sep[task_id])
        data_asc[task_id] = data[asc_id]
        taskcla.append((task_id,int(data[asc_id]['ncla'])))

    # Others
    n=0
    for t in data.keys():
        n+=data[t]['ncla']
    data['ncla']=n

    print('W2V AS')

    return data_asc,taskcla,vocab_size,embeddings


