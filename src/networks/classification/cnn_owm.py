import torch
import sys
import torch
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class Net(torch.nn.Module):

    def __init__(self, taskcla, args):
        super(Net, self).__init__()

        ncha = args.image_channel
        size = args.image_size

        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=2, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=False)

        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1000, bias=False)
        self.fc2 = torch.nn.Linear(1000, 1000, bias=False)

        torch.nn.init.xavier_normal(self.fc1.weight)
        torch.nn.init.xavier_normal(self.fc2.weight)

        self.args = args
        self.taskcla=taskcla

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(1000,args.nclasses)

        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(1000,n))

        print('DIL CNN')

        return


    def forward(self,x):
        output_dict = {}

        h_list = []
        x_list = []
        # Gated
        x = self.padding(x)
        x_list.append(torch.mean(x, 0, True))
        con1 = self.drop1(self.relu(self.c1(x)))
        con1_p = self.maxpool(con1)

        con1_p = self.padding(con1_p)
        x_list.append(torch.mean(con1_p, 0, True))
        con2 = self.drop1(self.relu(self.c2(con1_p)))
        con2_p = self.maxpool(con2)

        con2_p = self.padding(con2_p)
        x_list.append(torch.mean(con2_p, 0, True))
        con3 = self.drop1(self.relu(self.c3(con2_p)))
        con3_p = self.maxpool(con3)

        h = con3_p.view(x.size(0), -1)
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))

        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)
        output_dict['masks'] = None
        output_dict['x_list'] = x_list
        output_dict['h_list'] = h_list

        return output_dict

