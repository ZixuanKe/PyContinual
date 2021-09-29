import sys
import torch
import numpy as np
import torch.nn.functional as F

import utils
import torch.nn as nn
import math
import random
# Alextnet HAT from
# https://github.com/joansj/hat/blob/master/src/networks/alexnet_hat.py


class Net(torch.nn.Module):

    def __init__(self,taskcla,args):
        super(Net,self).__init__()

        ncha = args.image_channel
        size = args.image_size

        self.taskcla=taskcla
        self.args=args
        self.c1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.c2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.c3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*self.smid*self.smid,2048)
        self.fc2=torch.nn.Linear(2048,2048)

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),64)
        self.ec2=torch.nn.Embedding(len(self.taskcla),128)
        self.ec3=torch.nn.Embedding(len(self.taskcla),256)
        self.efc1=torch.nn.Embedding(len(self.taskcla),2048)
        self.efc2=torch.nn.Embedding(len(self.taskcla),2048)

        self.last=torch.nn.Linear(2048,args.nclasses)
        self.aem = AEM(2048,100,self.args.ntasks)



        print('DIL CNN HAT')

        return

    def forward(self,t,x,s=None):
        output_dict = {}


        out,id_detection,order_outputs = self.feature_extraction_contrast(x,t,s)
        output_dict['weighted_sum'] = out
        output_dict['order_outputs'] = order_outputs
        output_dict['weight'] = id_detection

        y = self.last(out)

        output_dict['y'] = y

        masks=self.mask(t,s=s)
        output_dict['masks'] = masks

        return output_dict

    def mix_output(self,t, x, s):
        output_dict = {}
        out,id_detection,order_outputs = self.feature_extraction_contrast(x,t,s)
        y = self.last(out)
        output_dict['y'] = y
        output_dict['masks'] = self.mask(t, s=s)

        return output_dict


    def feature_extraction_contrast(self,x,t,s):

        input_h = self.get_feature(x)
        domain_aware_hs = []
        for aem_t in range(self.args.ntasks):
            if aem_t == t:
                masks = self.mask(aem_t, s=s)
            else:
                with torch.no_grad():
                    masks = self.mask(aem_t, s=s)
                    masks = [mask.detach().clone() for mask in masks]

            domain_aware_h = self.get_feature_mask(x,masks)
            domain_aware_hs.append(domain_aware_h.unsqueeze(1))

        order_outputs = torch.cat(domain_aware_hs,dim=1)

        out, id_detection = self.aem(order_outputs,input_h,t)  # softmax on task
        return out,id_detection,order_outputs



    def get_feature_mask(self,x,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks

        h=self.maxpool(self.drop1(self.relu(self.c1(x))))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop1(self.relu(self.c2(h))))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop2(self.relu(self.c3(h))))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        return h

    def get_feature(self,x):
        h=self.maxpool(self.drop1(self.relu(self.c1(x))))
        h=self.maxpool(self.drop1(self.relu(self.c2(h))))
        h=self.maxpool(self.drop2(self.relu(self.c3(h))))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        return h

    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(torch.LongTensor([t]).cuda()))
        gc2=self.gate(s*self.ec2(torch.LongTensor([t]).cuda()))
        gc3=self.gate(s*self.ec3(torch.LongTensor([t]).cuda()))
        gfc1=self.gate(s*self.efc1(torch.LongTensor([t]).cuda()))
        gfc2=self.gate(s*self.efc2(torch.LongTensor([t]).cuda()))
        return [gc1,gc2,gc3,gfc1,gfc2]

    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks
        if n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc3.data.view(-1,1,1).expand((self.ec3.weight.size(1),self.smid,self.smid)).contiguous().view(1,-1).expand_as(self.fc1.weight)
            return torch.min(post,pre)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        elif n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2.weight)
            return torch.min(post,pre)
        elif n=='c2.bias':
            return gc2.data.view(-1)
        elif n=='c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3.weight)
            return torch.min(post,pre)
        elif n=='c3.bias':
            return gc3.data.view(-1)
        return None




class AEM(nn.Module):
    """ Self attention Layer:
    refer to https://github.com/huang50213/AIM-Fewshot-Continual/blob/a962291e4e24a03760c576b2fbc3d220cb029430/Continual/model/aim.py"""
    def __init__(self,input_dim,output_dim,num_units):
        super(AEM,self).__init__()
        self.num_input_heads = 1
        self.num_units = num_units
        self.input_value_size = output_dim
        self.hidden_size = input_dim
        self.input_size = input_dim
        self.input_query_size = input_dim
        self.input_key_size = input_dim

        self.hs_weight = nn.Parameter(
            math.sqrt(2. / self.input_size) * torch.randn(self.num_units, self.input_size, self.input_value_size))
        self.query_weight = nn.Parameter(math.sqrt(2. / self.hidden_size) * torch.randn(self.num_units, self.hidden_size,
                                                                                   self.input_query_size * self.num_input_heads))
        self.key_weight = nn.Parameter(torch.randn(self.num_input_heads * self.input_key_size, self.input_size))
        self.key_bias = nn.Parameter(torch.randn(self.num_input_heads * self.input_key_size))
        # value_weight = nn.Parameter(torch.randn(self.num_input_heads * self.input_value_size, self.input_size))
        # value_bias = nn.Parameter(torch.randn(self.num_input_heads * self.input_value_size))

        nn.init.normal_(self.key_weight, 0, math.sqrt(2. / self.input_size))
        nn.init.zeros_(self.key_bias)
        self.sigmoid=torch.nn.Sigmoid()


    def forward(self,domain_aware_h,input_h,t):

        input_h = input_h.unsqueeze(1)
        size = input_h.size()
        key_layer = F.linear(input_h, self.key_weight, self.key_bias)
        query_layer = self.grouplinearlayer(domain_aware_h, self.query_weight)

        # hs_value_layer = self.grouplinearlayer(input_h.unsqueeze(2).repeat(1, 1, self.num_units, 1).reshape(size[0], self.num_units, size[-1]), self.hs_weight)  # B*2 X num units X inval dim
        # hs_value_layer = hs_value_layer.reshape(size[0], 1, self.num_units, self.input_value_size).permute(0, 2, 1, 3) # B X num units X 2 X inval dim

        key_layer = self.transpose_for_scores(key_layer, self.num_input_heads, self.input_key_size)
        query_layer = self.transpose_for_scores(query_layer, self.num_input_heads, self.input_query_size)
        #key_layer = F.normalize(key_layer, p=2, dim=-1, eps=1e-12)
        #query_layer = F.normalize(query_layer, p=2, dim=-1, eps=1e-12)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.input_key_size)
        attention_scores = torch.mean(attention_scores, dim = 1)
        attention_scores = attention_scores.squeeze(-1)
        attention_scores = self.sigmoid(attention_scores)

        masks_ = torch.zeros_like(attention_scores)
        masks_[:,:t+1].fill_(1)
        attention_scores = attention_scores * masks_

        out = (domain_aware_h * ((attention_scores / attention_scores.sum(1,keepdim=True)).unsqueeze(-1))).sum(1)
        return out,attention_scores

    def grouplinearlayer(self, input, weight):
        input = input.permute(1, 0, 2)
        input = torch.bmm(input, weight)
        return input.permute(1, 0, 2)


    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)