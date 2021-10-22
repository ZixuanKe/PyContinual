# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import utils

class Shared(torch.nn.Module):

    def __init__(self,args):
        super(Shared, self).__init__()

        ncha = args.image_channel
        size = args.image_size
        self.num_tasks = args.ntasks


        self.conv1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*s*s,2048)
        self.fc2=torch.nn.Linear(2048,2048)


    def forward(self, x_s):
        x_s = x_s.view_as(x_s)
        h = self.maxpool(self.drop1(self.relu(self.conv1(x_s))))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h = h.view(x_s.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))
        return h



class Private(torch.nn.Module):
    def __init__(self, args):
        super(Private, self).__init__()

        ncha = args.image_channel
        size = args.image_size
        self.num_tasks = args.ntasks


        self.task_out = torch.nn.ModuleList()
        for _ in range(self.num_tasks):
            self.conv = torch.nn.Sequential()
            self.conv.add_module('conv1',torch.nn.Conv2d(ncha,64, kernel_size=size // 8))
            self.conv.add_module('relu1', torch.nn.ReLU(inplace=True))
            self.conv.add_module('drop1', torch.nn.Dropout(0.2))
            self.conv.add_module('maxpool1', torch.nn.MaxPool2d(2))
            s=utils.compute_conv_output_size(size,size//8)
            s=s//2
            self.conv.add_module('conv2', torch.nn.Conv2d(64,128, kernel_size=size // 10))
            self.conv.add_module('relu2', torch.nn.ReLU(inplace=True))
            self.conv.add_module('dropout2', torch.nn.Dropout(0.2))
            self.conv.add_module('maxpool2', torch.nn.MaxPool2d(2))
            s=utils.compute_conv_output_size(s,size//10)
            s=s//2
            self.conv.add_module('conv3',torch.nn.Conv2d(128,256,kernel_size=2))
            self.conv.add_module('relu3', torch.nn.ReLU(inplace=True))
            self.conv.add_module('dropout3', torch.nn.Dropout(0.5))
            self.conv.add_module('maxpool3', torch.nn.MaxPool2d(2))
            s=utils.compute_conv_output_size(s,2)
            s=s//2
            self.conv.add_module('maxpool2', torch.nn.MaxPool2d(2))
            self.task_out.append(self.conv)
            self.linear = torch.nn.Sequential()

            self.linear.add_module('linear1', torch.nn.Linear(256*s*s,2048))
            self.linear.add_module('relu3', torch.nn.ReLU(inplace=True))
            self.task_out.append(self.linear)


    def forward(self, x, t):
        x = x.view_as(x)
        out = self.task_out[2*t].forward(x)
        #TODO: check whether it is ok to use this in DIL. What is the different between use different head and different part

        out = out.view(out.size(0),-1)
        out = self.task_out[2*t+1].forward(out)
        return out



class Net(torch.nn.Module):

    def __init__(self, taskcla,args):
        super(Net, self).__init__()

        ncha = args.image_channel
        size = args.image_size
        self.taskcla = taskcla
        self.num_tasks = args.ntasks

        self.shared = Shared(args)
        self.private = Private(args)
        self.args = args

        if 'dil' in args.scenario:
            self.last= torch.nn.Sequential(
                        torch.nn.Linear(2*2048, 2048),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Dropout(),
                        torch.nn.Linear(2048, 2048),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(2048, self.args.nclasses)
                    )

        elif 'til' in args.scenario:
            self.last = torch.nn.ModuleList()
            for i in range(self.num_tasks):
                self.last.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(2*2048, 2048),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Dropout(),
                        torch.nn.Linear(2048, 2048),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Linear(2048, self.taskcla[i][1])
                    ))


    def forward(self, x_s, x_p, tt, t):
        output_dict = {}

        x_s = x_s.view_as(x_s)
        x_p = x_p.view_as(x_p)

        x_s = self.shared(x_s)
        x_p = self.private(x_p, t) # t decides which part of private to use

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
        print('Num parameters in P+p    = %s ' % self.pretty_print(count_P+count_H))
        print('-------------------------->   Architecture size: %s parameters (%sB)' % (self.pretty_print(count_S + count_P + count_H),
                                                                    self.pretty_print(4*(count_S + count_P + count_H))))

        print("-------------------------->   Memory size: %s samples per task (%sB)" % (self.samples,
                                                                                        self.pretty_print(self.num_tasks*4*self.samples*self.image_size)))
        print("------------------------------------------------------------------------------")
        print("                               TOTAL:  %sB" % self.pretty_print(4*(count_S + count_P + count_H)+self.num_tasks*4*self.samples*self.image_size))

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

