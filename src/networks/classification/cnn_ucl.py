import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear
from transformers import BertModel, BertConfig
import utils

class Net(nn.Module):
    def __init__(self, taskcla, args):
        super().__init__()

        ncha = args.image_channel
        size = args.image_size

        self.taskcla=taskcla

        self.conv1=BayesianConv2D(ncha,64,kernel_size=size//8,ratio=args.ratio)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=BayesianConv2D(64,128,kernel_size=size//10,ratio=args.ratio)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=BayesianConv2D(128,256,kernel_size=2,ratio=args.ratio)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.args = args

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=BayesianLinear(256*s*s,2048,ratio=args.ratio)
        self.fc2=BayesianLinear(2048,2048,ratio=args.ratio)
        self.old_weight_norm = []

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(2048,args.nclasses)
            self.merge_last=torch.nn.Linear(args.nclasses*2,args.nclasses)

        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            self.merge_last=torch.nn.ModuleList()

            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(2048,n))
                self.merge_last.append(torch.nn.Linear(n*2,n))

        print('DIL CNN')


    def forward(self,x, sample=False):
        output_dict = {}

        h=self.maxpool(self.drop1(self.relu(self.conv1(x,sample))))
        h=self.maxpool(self.drop1(self.relu(self.conv2(h,sample))))
        h=self.maxpool(self.drop2(self.relu(self.conv3(h,sample))))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h,sample)))
        h=self.drop2(self.relu(self.fc2(h,sample)))

        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)
        output_dict['masks'] = None

        return output_dict

