#coding: utf-8
import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()
        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)

        '''
        In case you want to fix some layers
        '''
        #BERT fixed some ===========
        # modules = [self.bert.embeddings, self.bert.encoder.layer[:args.activate_layer_num]] #Replace activate_layer_num by what you want
        # modules = [self.bert.encoder.layer[-1]]
        #
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False

        #BERT fixed all ===========
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.taskcla=taskcla
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(args.bert_hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(args.bert_hidden_size,n))


        print('DIL BERT')

        return

    def forward(self,input_ids, segment_ids, input_mask):
        output_dict = {}

        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        pooled_output = self.dropout(pooled_output)

        if 'dil' in self.args.scenario:
            y = self.last(pooled_output)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](pooled_output))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(pooled_output, dim=1)

        return output_dict