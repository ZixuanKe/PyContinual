import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self, taskcla,embeddings,args):

        super(Net,self).__init__()

        self.taskcla=taskcla
        self.args=args

        self.sentence_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float))
        self.aspect_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float))

        for param in self.sentence_embedding.parameters():
            param.requires_grad = False
        for param in self.aspect_embedding.parameters():
            param.requires_grad = False

        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM=[100, 100, 100]
        self.WORD_DIM = args.w2v_hidden_size

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

        print('W2V + CNN')

        return

    def forward(self, term,sentence):
        output_dict = {}

        sequence_output = self.sentence_embedding(sentence).float() #sentence only

        h = sequence_output.view(-1, 1, self.WORD_DIM * sentence.size(1))

        h1 = F.max_pool1d(F.relu(self.c1(h)), sentence.size(1) - self.FILTERS[0] + 1).view(-1, self.FILTER_NUM[0],1)
        h2 = F.max_pool1d(F.relu(self.c2(h)), sentence.size(1) - self.FILTERS[1] + 1).view(-1, self.FILTER_NUM[1],1)
        h3 = F.max_pool1d(F.relu(self.c3(h)), sentence.size(1) - self.FILTERS[2] + 1).view(-1, self.FILTER_NUM[2],1)

        h = torch.cat([h1,h2,h3], 1)
        h=h.view(sequence_output.size(0),-1)
        h = self.dropout(h)

        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))

        #loss ==============
        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)

        return output_dict