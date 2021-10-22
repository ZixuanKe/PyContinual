import sys
import torch

import utils
import torch.nn.functional as F
import numpy as np
# Alext net from
# https://github.com/joansj/hat/blob/master/src/networks/alexnet.py

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):
        super(Net,self).__init__()

        ncha = args.image_channel
        size = args.image_size

        self.taskcla=taskcla

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
        self.args = args


        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*s*s,2048)
        self.fc2=torch.nn.Linear(2048,2048)
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

        print('CNN')


        return

    def forward(self,x):
        output_dict = {}

        h = self.features(x)

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

    def features(self,x):
        h=self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h=self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h=self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        return h




    def save_old_norm(self):
        old_weight = self.last.weight.cpu().detach().numpy().copy
        self.old_weight_norm.append(old_weight)

    def align_norms(self):
        # Fetch old and new layers

        new_layer = self.last
        old_layers = self.old_weight_norm

        # Get weight of layers
        new_weight = new_layer.weight.cpu().detach().numpy()

        old_weight = np.concatenate([old_layers[i] for i in range(len(self.old_weight_norm))])

        print("old_weight's shape is: ",old_weight.shape)
        print("new_weight's shape is: ",new_weight.shape)

        # Calculate the norm
        Norm_of_new = np.linalg.norm(new_weight, axis=1)
        Norm_of_old = np.linalg.norm(old_weight, axis=1)
        # Calculate the Gamma
        gamma = np.mean(Norm_of_new) / np.mean(Norm_of_old)
        print("Gamma = ", gamma)

        # Update new layer's weight
        updated_new_weight = torch.Tensor(gamma * new_weight).cuda()
        self.last.weight = torch.nn.Parameter(updated_new_weight)



    def merge_head(self,h):
        if 'dil' in self.args.scenario:
            y = self.merge_last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.merge_last[t](h))
        return y



    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                    + 100 * output_size + output_size)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (input_size * 100
                    + 100 + 100 * 100 + 100 + 100 * output_size + output_size)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def num_flat_features(self, x: torch.Tensor) -> int:
        """
        Computes the total number of items except the first dimension.
        :param x: input tensor
        :return: number of item from the second dimension onward
        """
        size = x.size()[1:]
        num_features = 1
        for ff in size:
            num_features *= ff
        return num_features