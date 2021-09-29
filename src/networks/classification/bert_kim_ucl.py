import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear
from transformers import BertModel, BertConfig

class Net(nn.Module):
    def __init__(self, taskcla, args):
        super().__init__()

        self.args=args
        self.taskcla = taskcla
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM=[100, 100, 100]
        self.WORD_DIM = args.bert_hidden_size
        self.MAX_SENT_LEN = args.max_seq_length

        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False

        self.bert = BertModel.from_pretrained(args.bert_model,config=config)

        #BERT fixed, i.e. BERT as feature extractor===========
        for param in self.bert.parameters():
            param.requires_grad = False


        self.convs = nn.ModuleList([BayesianConv2D(1, 100, (K, self.WORD_DIM)) for K in self.FILTERS])

        self.dropout=torch.nn.Dropout(0.5)

        if 'dil' in self.args.scenario:
            self.last=torch.nn.Linear(sum(self.FILTER_NUM),self.args.nclasses)
        elif 'til' in self.args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(sum(self.FILTER_NUM),n))



    def forward(self, input_ids, segment_ids, input_mask, sample=False):
        output_dict = {}

        sequence_output, pooled_output = \
          self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        h = sequence_output.view(-1, 1, self.args.max_seq_length,self.WORD_DIM)

        h = [F.relu(conv(h,sample)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        h = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]  # [(N, Co), ...]*len(Ks)
        h = torch.cat(h, 1)

        h = self.dropout(h)

        if 'dil' in self.args.scenario:
            y=self.last(h)

        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))
        output_dict['y'] = y
        return output_dict

