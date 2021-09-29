import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()

        self.taskcla=taskcla
        self.args=args

        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)

        #BERT fixed, i.e. BERT as feature extractor===========
        for param in self.bert.parameters():
            param.requires_grad = False

        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM=[100, 100, 100]
        self.WORD_DIM = args.bert_hidden_size

        self.relu=torch.nn.ReLU()

        self.c1 = torch.nn.Conv1d(1, self.FILTER_NUM[0], self.WORD_DIM * self.FILTERS[0], stride=self.WORD_DIM)
        self.c2 = torch.nn.Conv1d(1, self.FILTER_NUM[1], self.WORD_DIM * self.FILTERS[1], stride=self.WORD_DIM)
        self.c3 = torch.nn.Conv1d(1, self.FILTER_NUM[2], self.WORD_DIM * self.FILTERS[2], stride=self.WORD_DIM)

        self.dropout = nn.Dropout(0.5)

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(sum(self.FILTER_NUM),args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(sum(self.FILTER_NUM),n))

        print(' BERT (Fixed) + CNN')

        return

    def forward(self,input_ids, segment_ids, input_mask):
        output_dict = {}

        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        #sentence ============
        h = sequence_output.view(-1, 1, self.WORD_DIM * self.args.max_seq_length)

        h1 = F.max_pool1d(F.relu(self.c1(h)), self.args.max_seq_length - self.FILTERS[0] + 1).view(-1, self.FILTER_NUM[0],1)
        h2 = F.max_pool1d(F.relu(self.c2(h)), self.args.max_seq_length - self.FILTERS[1] + 1).view(-1, self.FILTER_NUM[1],1)
        h3 = F.max_pool1d(F.relu(self.c3(h)), self.args.max_seq_length - self.FILTERS[2] + 1).view(-1, self.FILTER_NUM[2],1)

        h = torch.cat([h1,h2,h3], 1)
        h=h.view(sequence_output.size(0),-1)
        h = self.dropout(h)

        #loss ==============

        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)

        return output_dict
