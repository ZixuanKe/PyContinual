#Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import BertTokenizer as BertTokenizer
import os
import torch
import numpy as np
import random
import nlp_data_utils as data_utils
from nlp_data_utils import ABSATokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import math



# domains = ['Video_Games','Toys_and_Games','Tools_and_Home_Improvement','Sports_and_Outdoors','Pet_Supplies',
#            'Patio_Lawn_and_Garden','Office_Products','Musical_Instruments','Movies_and_TV',
#            'Kindle_Store','Home_and_Kitchen','Health_and_Personal_Care','Grocery_and_Gourmet_Food','Electronics',
#            'Digital_Music','Clothing_Shoes_and_Jewelry','Cell_Phones_and_Accessories','CDs_and_Vinyl',
#            'Books','Beauty','Baby','Automotive','Apps_for_Android','Amazon_Instant_Video']

domains = ['airconditioner','bike','diaper','GPS','headphone',
           'hotel','luggage','smartphone','stove','TV']

datasets = [ './dat/ssc/'+domain for domain in domains]

def get(logger=None,args=None):
    data={}
    taskcla=[]

    # Others
    f_name = 'ssc_random'

    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    print('random_sep: ',random_sep)
    print('domains: ',domains)


    for t in range(args.ntasks):
        dataset = datasets[domains.index(random_sep[t])]

        data[t]={}
        data[t]['name']=dataset
        data[t]['ncla']=3

        print('dataset: ',dataset)

        processor = data_utils.DscProcessor()
        label_list = processor.get_labels()
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        train_examples = processor.get_train_examples(dataset)
        if args.train_data_size > 0:
            random.Random(args.seed).shuffle(train_examples) #more robust
            train_examples = train_examples[:args.train_data_size]

        num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs
        # num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

        train_features = data_utils.convert_examples_to_features_dsc(
            train_examples, label_list, args.max_seq_length, tokenizer, "dsc")
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)

        data[t]['train'] = train_data
        data[t]['num_train_steps']=num_train_steps

        valid_examples = processor.get_dev_examples(dataset)
        if args.dev_data_size > 0:
            random.Random(args.seed).shuffle(valid_examples) #more robust
            valid_examples = valid_examples[:args.dev_data_size]

        valid_features=data_utils.convert_examples_to_features_dsc(
            valid_examples, label_list, args.max_seq_length, tokenizer, "dsc")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_tasks = torch.tensor([t for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids, valid_all_tasks)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        data[t]['valid']=valid_data


        processor = data_utils.DscProcessor()
        label_list = processor.get_labels()
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        eval_examples = processor.get_test_examples(dataset)

        eval_features = data_utils.convert_examples_to_features_dsc(eval_examples, label_list, args.max_seq_length, tokenizer, "dsc")

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)
        # Run prediction for full data

        data[t]['test']=eval_data

        taskcla.append((t,int(data[t]['ncla'])))



    # Others
    n=0
    for t in data.keys():
        n+=data[t]['ncla']
    data['ncla']=n


    return data,taskcla


