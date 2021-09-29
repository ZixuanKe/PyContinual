import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F
import math
# Reproduce the SRK according to paper
# Sentiment-Classification-by-Leveraging-the-Shared-Knowledge-from-a-Sequence-of-Domains.pdf
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


        self.gate=torch.nn.Sigmoid()
        self.relu=torch.nn.ReLU()
        self.fln = FLN(args,taskcla)
        self.krn = KRN(args,taskcla)


        self.fln_fc= torch.nn.Linear(args.w2v_hidden_size,args.w2v_hidden_size,bias=False)
        self.krn_fc= torch.nn.Linear(args.w2v_hidden_size,args.w2v_hidden_size,bias=False)




        if 'dil' in args.scenario:
            self.last = torch.nn.Linear(args.w2v_hidden_size,args.nclasses)
            self.krn_last = torch.nn.Linear(args.w2v_hidden_size,args.nclasses)
            self.fln_last = torch.nn.Linear(args.w2v_hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            self.krn_last=torch.nn.ModuleList()
            self.fln_last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(args.w2v_hidden_size,n))
                self.krn_last.append(torch.nn.Linear(args.w2v_hidden_size,n))
                self.fln_last.append(torch.nn.Linear(args.w2v_hidden_size,n))

        print('W2V (Fixed) + GRU + SRK')


        return

    def forward(self,t, term,sentence):
        output_dict = {}

        sequence_output = self.sentence_embedding(sentence).float() #sentence only

        fln_output,fln_hidden,c1,c2,c3 = self.fln.gru(sequence_output)
        krn_output,krn_hidden,control_1,control_2,control_3 = self.krn.gru(sequence_output)

        g=self.gate(self.fln_fc(fln_hidden) + self.krn_fc(krn_hidden))
        h = (1-g)*fln_hidden+ g*(krn_hidden)

        #loss ==============

        if 'dil' in self.args.scenario:
            y = self.last(h)
            krn_y = self.last(krn_hidden)
            fln_y = self.last(fln_hidden)

        elif 'til' in self.args.scenario:
            y=[]
            krn_y=[]
            fln_y=[]

            for t,i in self.taskcla:
                y.append(self.last[t](h))
                krn_y.append(self.last[t](krn_hidden))
                fln_y.append(self.last[t](fln_hidden))

        output_dict['y'] = y
        output_dict['fln_y'] = fln_y
        output_dict['krn_y'] = krn_y
        output_dict['control_1'] = control_1.data.clone()
        output_dict['control_2'] = control_2.data.clone()
        output_dict['control_3'] = control_3.data.clone()

        return output_dict

    def get_view_for(self,n,control_1_mask,control_2_mask,control_3_mask):
        if n=='krn.gru.gru_cell.x2h.weight':
            # print('not none')
            return control_1_mask.data.view(1,-1).expand_as(self.krn.gru.gru_cell.x2h.weight)
        elif n=='krn.gru.gru_cell.h2h.weight':
            return control_2_mask.data.view(1,-1).expand_as(self.krn.gru.gru_cell.h2h.weight)
        elif n=='krn.gru.gru_cell.h2h.bias':
            return torch.cat([control_1_mask,control_2_mask,control_3_mask])
        elif n=='krn.gru.gru_cell.x2h.bias':
            return torch.cat([control_1_mask,control_2_mask,control_3_mask])
        return None


class FLN(nn.Module):
    def __init__(self,args,taskcla):
        super().__init__()

        # self.gru = GRU(
        #             embedding_dim = args.w2v_hidden_size,
        #             hidden_dim = args.w2v_hidden_size,
        #             n_layers=1,
        #             bidirectional=False,
        #             dropout=0.5,
        #             args=args)

        self.gru = GRUModel(
                    input_dim = args.w2v_hidden_size,
                    hidden_dim = args.w2v_hidden_size,
                    layer_dim=args.w2v_hidden_size,
                    output_dim=args.w2v_hidden_size)


class KRN(nn.Module):
    def __init__(self,args,taskcla):
        super().__init__()

        # self.gru = GRU(
        #             embedding_dim = args.w2v_hidden_size,
        #             hidden_dim = args.w2v_hidden_size,
        #             n_layers=1,
        #             bidirectional=False,
        #             dropout=0.5,
        #             args=args)

        self.gru = GRUModel(
                    input_dim = args.w2v_hidden_size,
                    hidden_dim = args.w2v_hidden_size,
                    layer_dim=args.w2v_hidden_size,
                    output_dim=args.w2v_hidden_size)



class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        outs = []
        control_1 = []
        control_2 = []
        control_3 = []
        hn = h0[0,:,:]
        for seq in range(x.size(1)):
            hn,c_1,c_2,c_3 = self.gru_cell(x[:,seq,:], hn)
            outs.append(hn)
            control_1.append(c_1)
            control_2.append(c_2)
            control_3.append(c_3)

        output = torch.cat(outs)
        control_1 = torch.cat(control_1)
        control_2 = torch.cat(control_2)
        control_3 = torch.cat(control_3)

        hidden = outs[-1].squeeze()

        if len(hidden.size()) < 2:
            hidden = hidden.unsqueeze(0)

        return output,hidden,control_1,control_2,control_3


class GRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
        self.k = 0.001
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):

        x = x.view(-1, x.size(1))
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        if len(gate_x.size()) < 2:
            gate_x = gate_x.unsqueeze(0)
            gate_h = gate_h.unsqueeze(0)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)


        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = i_n + (resetgate * h_n)
        newgate[newgate>=0.9] = 0.9+self.k*newgate[newgate>=0.9]
        newgate[newgate<=0] = self.k*newgate[newgate<=0]

        hy = newgate + inputgate * (hidden - newgate)


        return hy,x,hidden,(resetgate * h_n)


# class GRU(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, n_layers,
#                  bidirectional, dropout, args):
#         super().__init__()
#
#         self.rnn = nn.GRU(embedding_dim,
#                            hidden_dim,
#                            num_layers=n_layers,
#                            bidirectional=bidirectional,
#                            dropout=dropout,
#                            batch_first=True)
#         self.args = args
#
#     def forward(self, x):
#         output, hidden = self.rnn(x)
#         hidden = hidden.view(-1,self.args.w2v_hidden_size)
#         output = output.view(-1,self.args.max_seq_length,self.args.w2v_hidden_size)
#
#         return output,hidden

