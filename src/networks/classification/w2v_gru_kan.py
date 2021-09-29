import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,taskcla,embeddings,args):

        super(Net,self).__init__()

        self.taskcla=taskcla
        self.args=args

        self.sentence_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float))
        self.aspect_embedding = torch.nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float))

        for param in self.sentence_embedding.parameters():
            param.requires_grad = False
        for param in self.aspect_embedding.parameters():
            param.requires_grad = False


        self.relu=torch.nn.ReLU()
        self.mcl = MCL(args,taskcla)
        self.ac = AC(args,taskcla)

        self.last=torch.nn.ModuleList()

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(args.w2v_hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(args.w2v_hidden_size,n))

        print('W2V (Fixed) + GRU + KAN')


        return

    def forward(self,t, term,sentence, which_type,s):
        output_dict = {}

        sequence_output = self.sentence_embedding(sentence).float() #sentence only

        gfc=self.ac.mask(t=t,s=s)

        if which_type == 'mcl':
            mcl_output,mcl_hidden = self.mcl.gru(sequence_output)
            if t == 0: mcl_hidden = mcl_hidden*torch.ones_like(gfc.expand_as(mcl_hidden)) # everyone open
            else: mcl_hidden=mcl_hidden*gfc.expand_as(mcl_hidden)

            h=self.relu(mcl_hidden)

        elif which_type == 'ac':
            mcl_output,mcl_hidden = self.mcl.gru(sequence_output)
            mcl_output=self.relu(mcl_output)
            mcl_output=mcl_output*gfc.expand_as(mcl_output)
            ac_output,ac_hidden = self.ac.gru(mcl_output)
            h=self.relu(ac_hidden)

        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)

        return output_dict

    def get_view_for(self,n,mask):
        if n=='mcl.gru.rnn.weight_ih_l0':
            # print('not none')
            return mask.data.view(1,-1).expand_as(self.mcl.gru.rnn.weight_ih_l0)
        elif n=='mcl.gru.rnn.weight_hh_l0':
            return mask.data.view(1,-1).expand_as(self.mcl.gru.rnn.weight_hh_l0)
        elif n=='mcl.gru.rnn.bias_ih_l0':
            return mask.data.view(-1).repeat(3)
        elif n=='mcl.gru.rnn.bias_hh_l0':
            return mask.data.view(-1).repeat(3)
        return None



class AC(nn.Module):
    def __init__(self,args,taskcla):
        super().__init__()

        self.gru = GRU(
                    embedding_dim = args.w2v_hidden_size,
                    hidden_dim = args.w2v_hidden_size,
                    n_layers=1,
                    bidirectional=False,
                    dropout=0.5,
                    args=args)

        self.efc=torch.nn.Embedding(args.ntasks,args.w2v_hidden_size)
        self.gate=torch.nn.Sigmoid()


    def mask(self,t,s=1):
        gfc=self.gate(s*self.efc(torch.LongTensor([t]).cuda()))
        return gfc


class MCL(nn.Module):
    def __init__(self,args,taskcla):
        super().__init__()

        self.gru = GRU(
                    embedding_dim = args.w2v_hidden_size,
                    hidden_dim = args.w2v_hidden_size,
                    n_layers=1,
                    bidirectional=False,
                    dropout=0.5,
                    args=args)

class GRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers,
                 bidirectional, dropout, args):
        super().__init__()

        self.rnn = nn.GRU(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        self.args = args

    def forward(self, x):
        output, hidden = self.rnn(x)
        hidden = hidden.view(-1,self.args.w2v_hidden_size)
        output = output.view(-1,x.size(1),self.args.w2v_hidden_size)
        # output = output.view(-1,self.args.max_sentence_length,self.args.w2v_hidden_size)

        return output,hidden