import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear
from transformers import BertModel, BertConfig

class Net(nn.Module):
    def __init__(self, taskcla,embeddings,ratio, args):
        super().__init__()

        self.args=args
        self.taskcla = taskcla
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM=[100, 100, 100]
        self.WORD_DIM = args.w2v_hidden_size
        self.MAX_SENT_LEN = args.max_sentence_length

        self.sentence_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float))
        self.aspect_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float))

        for param in self.sentence_embedding.parameters():
            param.requires_grad = False
        for param in self.aspect_embedding.parameters():
            param.requires_grad = False


        self.convs = nn.ModuleList([BayesianConv2D(1, 100, (K, self.WORD_DIM)) for K in self.FILTERS])

        self.dropout=torch.nn.Dropout(0.5)

        self.last=torch.nn.ModuleList()

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(sum(self.FILTER_NUM),args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(sum(self.FILTER_NUM),n))


        print('W2V + CNN UCL')

    def forward(self, term,sentence, sample=False):
        output_dict = {}

        sequence_output = self.sentence_embedding(sentence).float() #sentence only

        h = sequence_output.view(-1, 1, sentence.size(1),self.WORD_DIM)

        h = [F.relu(conv(h,sample)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        h = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]  # [(N, Co), ...]*len(Ks)
        h = torch.cat(h, 1)


        h = self.dropout(h)

        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)

        return output_dict



