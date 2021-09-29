#Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import BertTokenizer as BertTokenizer
import os
import torch
import numpy as np
import sys
import random
import nlp_data_utils as data_utils
from nlp_data_utils import ABSATokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import math
sys.path.append(os.path.abspath('./dataloaders'))
from contrastive_dataset import InstanceSample


datasets = [
            './dat/nli/travel',
            './dat/nli/telephone',
            './dat/nli/slate',
            './dat/nli/government',
            './dat/nli/fiction'
            ]


domains = [
     'travel',
     'telephone',
     'slate',
     'government',
     'fiction']



def get(logger=None,args=None):
    data={}
    taskcla=[]

    # Others
    f_name = 'nli_random'

    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    print('random_sep: ',random_sep)
    print('domains: ',domains)


    for t in range(args.ntasks):
        dataset = datasets[domains.index(random_sep[t])]

        data[t]={}
        data[t]['name']=dataset
        data[t]['ncla']=3 #'neutral': 0, 'entailment': 1, 'contradiction': 2

        print('dataset: ',dataset)

        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        train_examples = processor.get_train_examples(dataset)
        if args.train_data_size > 0:
            random.Random(args.data_seed).shuffle(train_examples)
            train_examples = train_examples[:args.train_data_size]

        #Doble Saved: So that no change on data even if we change the seed
        # if args.train_data_size > 0:
        #     if not os.path.exists("./dat/nli/"+str(random_sep[t])+"/train_data_"+str(args.train_data_size)+'_'+str(args.data_seed)):
        #         torch.save(train_examples,"./dat/nli/"+str(random_sep[t])+"/train_data_"+str(args.train_data_size)+'_'+str(args.data_seed))
        #         logger.info("  save")
        #     else: #in that case, change seed will not change data
        #         train_examples = torch.load("./dat/nli/"+str(random_sep[t])+"/train_data_"+str(args.train_data_size)+'_'+str(0)) #always use 0
        #         logger.info("  load")

        #how many originally?
        # dataset:  ./dat/nli/telephone
        # 02/25/2021 16:25:42 - INFO - __main__ -   ***** Running training *****
        # 02/25/2021 16:25:42 - INFO - __main__ -     Num examples = 75013
        # 02/25/2021 16:25:42 - INFO - __main__ -     Batch size = 32
        # 02/25/2021 16:25:42 - INFO - __main__ -     Num steps = 46900
        # 02/25/2021 16:25:47 - INFO - __main__ -   ***** Running validations *****
        # 02/25/2021 16:25:47 - INFO - __main__ -     Num orig examples = 8335
        # 02/25/2021 16:25:47 - INFO - __main__ -     Num split examples = 8335
        # 02/25/2021 16:25:47 - INFO - __main__ -     Batch size = 32
        # 02/25/2021 16:25:48 - INFO - __main__ -   ***** Running evaluation *****
        # 02/25/2021 16:25:48 - INFO - __main__ -     Num examples = 1966
        # 02/25/2021 16:25:48 - INFO - __main__ -     Batch size = 128


        num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs
        # num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs
        train_features = data_utils.convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, "nli")
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)

        if args.distill_loss:
            train_data = InstanceSample(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)
        else:
            train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)

        data[t]['train'] = train_data
        data[t]['num_train_steps']=num_train_steps

        valid_examples = processor.get_dev_examples(dataset)
        if args.dev_data_size > 0:
            random.Random(args.data_seed).shuffle(valid_examples)
            valid_examples = valid_examples[:args.dev_data_size]

        # Doble Saved:  So that no change on data even if we change the seed
        # if args.dev_data_size > 0:
        #     if not os.path.exists("./dat/nli/"+str(random_sep[t])+"/valid_data_"+str(args.dev_data_size)+'_'+str(args.data_seed)):
        #         torch.save(valid_examples,"./dat/nli/"+str(random_sep[t])+"/valid_data_"+str(args.dev_data_size)+'_'+str(args.data_seed))
        #         logger.info("  save")
        #     else: #in that case, change seed will not change data
        #         valid_examples = torch.load("./dat/nli/"+str(random_sep[t])+"/valid_data_"+str(args.dev_data_size)+'_'+str(0)) #always use 0
        #         logger.info("  load")


        valid_features=data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, "nli")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_tasks = torch.tensor([t for f in valid_features], dtype=torch.long)

        if args.distill_loss:
            valid_data = InstanceSample(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids, valid_all_tasks)
        else:
            valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids, valid_all_tasks)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)


        data[t]['valid']=valid_data


        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        eval_examples = processor.get_test_examples(dataset)


        eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, "nli")

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)

        if args.distill_loss:
            eval_data = InstanceSample(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)
        else:
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


