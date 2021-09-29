import sys
import torch

import utils
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,taskcla,args=0):
        super(Net,self).__init__()

        ncha = args.image_channel
        size = args.image_size
        nhid = args.mlp_adapter_size

        self.taskcla=taskcla

        pdrop1=0.2
        pdrop2=0.5

        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)

        self.fc1=torch.nn.Linear(ncha*size*size,nhid)
        self.fc2=torch.nn.Linear(nhid,nhid)

        self.args = args

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(nhid,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(nhid,n))


        print('MLP')

        return

    def forward(self,x):
        output_dict = {}
        h_list = []

        h=self.drop1(x.view(x.size(0),-1))
        h_list.append(torch.mean(h, 0, True))

        h=self.drop2(self.relu(self.fc1(h)))
        h_list.append(torch.mean(h, 0, True))

        h=self.drop2(self.relu(self.fc2(h)))

        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)
        output_dict['masks'] = None
        output_dict['h_list'] = h_list
        output_dict['x_list'] = None

        return output_dict
