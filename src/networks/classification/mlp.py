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
        h=self.drop1(x.view(x.size(0),-1))
        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        return h


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