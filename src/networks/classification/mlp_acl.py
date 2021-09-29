# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class Private(torch.nn.Module):
    def __init__(self, args):
        super(Private, self).__init__()

        ncha = args.image_channel
        size = args.image_size
        nhid = args.mlp_adapter_size

        self.num_tasks = args.ntasks

        self.task_out = torch.nn.ModuleList()
        for _ in range(self.num_tasks):
            self.linear = torch.nn.Sequential()
            self.linear.add_module('linear', torch.nn.Linear(ncha*size*size, nhid))
            self.linear.add_module('relu', torch.nn.ReLU(inplace=True))
            self.task_out.append(self.linear)

    def forward(self, x_p, t):
        x_p = x_p.view(x_p.size(0), -1)
        return self.task_out[t].forward(x_p)



class Shared(torch.nn.Module):

    def __init__(self,args):
        super(Shared, self).__init__()

        ncha = args.image_channel
        size = args.image_size
        nhid = args.mlp_adapter_size

        self.relu=torch.nn.ReLU()
        self.drop=torch.nn.Dropout(0.2)
        self.fc1=torch.nn.Linear(ncha*size*size, nhid)

        self.fc2 = torch.nn.Linear(nhid,nhid)

    def forward(self, x_s):

        h = x_s.view(x_s.size(0), -1)
        h = self.drop(self.relu(self.fc1(h)))
        h = self.drop(self.relu(self.fc2(h)))

        return h


class Net(torch.nn.Module):

    def __init__(self, taskcla,args):
        super(Net, self).__init__()

        ncha = args.image_channel
        size = args.image_size
        nhid = args.mlp_adapter_size

        self.taskcla=taskcla
        self.num_tasks = args.ntasks


        self.shared = Shared(args)
        self.private = Private(args)
        self.args = args

        if 'dil' in args.scenario:
            self.last= torch.nn.Sequential(
                        torch.nn.Linear(2 * nhid, nhid),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Dropout(),
                        torch.nn.Linear(nhid, nhid),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(nhid, self.args.nclasses))

        elif 'til' in args.scenario:
            self.last = torch.nn.ModuleList()
            for i in range(self.num_tasks):
                self.last.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(2 * nhid, nhid),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Dropout(),
                        torch.nn.Linear(nhid, nhid),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(nhid, self.taskcla[i][1])
                    ))

        print('MLP ACL')

    def forward(self,x_s, x_p, tt, t):
            # train_tt = []  # task module labels
            # train_td = []  # disctiminator labels

        output_dict = {}

        h_s = x_s.view(x_s.size(0), -1)
        h_p = x_s.view(x_p.size(0), -1)

        x_s = self.shared(h_s)
        x_p = self.private(h_p, t)

        x = torch.cat([x_p, x_s], dim=1)

        if 'dil' in self.args.scenario:
            y = self.last(x)

        if 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](x))


        output_dict['y'] = y
        return output_dict

    def get_encoded_ftrs(self, x_s, x_p, t):
        return self.shared(x_s), self.private(x_p, t)


    def print_model_size(self):
        count_P = sum(p.numel() for p in self.private.parameters() if p.requires_grad)
        count_S = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
        count_H = sum(p.numel() for p in self.last.parameters() if p.requires_grad)

        print('Num parameters in S       = %s ' % (self.pretty_print(count_S)))
        print('Num parameters in P       = %s,  per task = %s ' % (self.pretty_print(count_P),self.pretty_print(count_P/self.num_tasks)))
        print('Num parameters in p       = %s,  per task = %s ' % (self.pretty_print(count_H),self.pretty_print(count_H/self.num_tasks)))
        print('Num parameters in P+p     = %s ' % self.pretty_print(count_P+count_H))
        print('-------------------------->   Total architecture size: %s parameters (%sB)' % (self.pretty_print(count_S + count_P + count_H),
                                                                    self.pretty_print(4*(count_S + count_P + count_H))))

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
