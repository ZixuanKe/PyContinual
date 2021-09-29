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

        self.c1 = torch.nn.Conv1d(1, self.FILTER_NUM[0], self.WORD_DIM * self.FILTERS[0], stride=self.WORD_DIM)
        self.c2 = torch.nn.Conv1d(1, self.FILTER_NUM[1], self.WORD_DIM * self.FILTERS[1], stride=self.WORD_DIM)
        self.c3 = torch.nn.Conv1d(1, self.FILTER_NUM[2], self.WORD_DIM * self.FILTERS[2], stride=self.WORD_DIM)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(300,args.bert_hidden_size)
        self.fc2 = nn.Linear(args.bert_hidden_size,args.bert_hidden_size)

        self.efc1=torch.nn.Embedding(len(self.taskcla),args.bert_hidden_size)
        self.efc2=torch.nn.Embedding(len(self.taskcla),args.bert_hidden_size)
        self.ec1=torch.nn.Embedding(len(self.taskcla),self.FILTER_NUM[0])
        self.ec2=torch.nn.Embedding(len(self.taskcla),self.FILTER_NUM[1])
        self.ec3=torch.nn.Embedding(len(self.taskcla),self.FILTER_NUM[2])

        self.gate=torch.nn.Sigmoid()

        self.relu=torch.nn.ReLU()

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(args.bert_hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(args.bert_hidden_size,n))

        print('CONTEXTUAL + KIM + HAT')

        return

    def forward(self,t,input_ids, segment_ids, input_mask,s):

        output_dict = {}

        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gfc1,gfc2=masks

        h = sequence_output.view(-1, 1, self.WORD_DIM * self.args.max_seq_length)

        h1 = F.max_pool1d(F.relu(self.c1(h)), self.args.max_seq_length - self.FILTERS[0] + 1).view(-1, self.FILTER_NUM[0],1)
        h1 = h1*gc1.unsqueeze(-1).expand_as(h1)

        h2 = F.max_pool1d(F.relu(self.c2(h)), self.args.max_seq_length - self.FILTERS[1] + 1).view(-1, self.FILTER_NUM[1],1)
        h2 = h2*gc2.unsqueeze(-1).expand_as(h2)

        h3 = F.max_pool1d(F.relu(self.c3(h)), self.args.max_seq_length - self.FILTERS[2] + 1).view(-1, self.FILTER_NUM[2],1)
        h3 = h3*gc3.unsqueeze(-1).expand_as(h3)

        h = torch.cat([h1,h2,h3], 1)
        h=h.view(sequence_output.size(0),-1)



        h=self.dropout(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)

        h=self.dropout(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)

        #loss ==============
        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))

        output_dict['y'] = y
        output_dict['masks'] = masks
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)

        return output_dict

    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        return [gc1,gc2,gc3,gfc1,gfc2]


    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks

        if n=='c1.weight':
            return gc1.data.view(-1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)

        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1).expand_as(self.c2.weight)
            return post
        elif n=='c2.bias':
            return gc2.data.view(-1)

        elif n=='c3.weight':
            post=gc3.data.view(-1,1,1).expand_as(self.c3.weight)
            return post
        elif n=='c3.bias':
            return gc3.data.view(-1)

        elif n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc3.data.view(-1,1).expand((self.ec3.weight.size(1),3)).contiguous().view(1,-1).expand_as(self.fc1.weight)
            return torch.min(post,pre)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)


        return None