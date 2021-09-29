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

        self.convs = nn.ModuleList([ torch.nn.Conv2d(1, 100, (K, self.WORD_DIM)) for K in self.FILTERS])

        self.fc1 = torch.nn.Linear(300, 300, bias=False)
        self.fc2 = torch.nn.Linear(300, 300, bias=False)

        self.dropout = nn.Dropout(0.5)


        self.last=torch.nn.ModuleList()

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(sum(self.FILTER_NUM),args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(sum(self.FILTER_NUM),n))


        print('W2V + CNN OWM')

        return

    def forward(self, term,sentence):
        output_dict = {}

        x_list = []
        h_list = []

        sequence_output = self.sentence_embedding(sentence).float() #sentence only

        h = sequence_output.view(-1, 1,sentence.size(1),self.WORD_DIM)
        x_list.append(torch.mean(h, 0, True))

        h = [F.relu(conv(h)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        h = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h]  # [(N, Co), ...]*len(Ks)
        x_list += [torch.mean(h_e, 0, True) for h_e in h]

        h = torch.cat(h, 1)

        h = self.relu(self.fc1(h))
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))
        h_list.append(torch.mean(h, 0, True))

        h = self.dropout(h)

        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)
        output_dict['x_list'] = x_list
        output_dict['h_list'] = h_list

        return output_dict
