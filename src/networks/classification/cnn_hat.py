import sys
import torch
import numpy as np
import torch.nn.functional as F

import utils
import torch.nn as nn
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

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(2048,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(2048,n))

        if 'Attn-HCHP-Outside' in self.args.mix_type:
            if self.args.task_based:
                self.self_attns = nn.ModuleList()
                for t in range(args.ntasks):
                    self.self_attns_ = nn.ModuleList()
                    offset = 0
                    for n in range(args.naug):
                        if n > 1: offset+=1
                        if t+1-offset==0: break
                        self.self_attns_.append(Self_Attn(t+1-offset))
                    self.self_attns.append(self.self_attns_)

        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""

        print('DIL CNN HAT')

        return

    def forward(self,t,x,start_mixup=None,s=None,l=None,idx=None,mix_type=None):
        output_dict = {}


        if start_mixup and  'Attn-HCHP-Outside' in mix_type:
            # print('attn type: ', self.args.attn_type)

            masks=self.mask(t=t,s=s)
            gc1,gc2,gc3,gfc1,gfc2=masks
            h=self.get_feature(x,gc1,gc2,gc3,gfc1,gfc2)
            if self.args.attn_type == 'self':
                h = self.self_attention_feature(t,x,h,l,idx,self.args.smax)

        else:
            # print('others: ')

            masks=self.mask(t,s=s)
            gc1,gc2,gc3,gfc1,gfc2=masks
            h=self.get_feature(x,gc1,gc2,gc3,gfc1,gfc2)

        output_dict['masks'] = masks
        output_dict['normalized_pooled_rep'] = F.normalize(h, dim=1)

        if 'dil' in self.args.scenario:
            y = self.last(h)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](h))

        output_dict['y'] = y

        return output_dict

    def self_attention_feature(self,t,x,pooled_output,order,idx,smax):
        if self.args.feature_based:
            pre_hs = []
            for pre_t in order:
                with torch.no_grad():
                    gc1,gc2,gc3,gfc1,gfc2=self.mask(t=pre_t,s=smax)
                    pre_h = \
                        self.get_feature(x,gc1,gc2,gc3,gfc1,gfc2)
                pre_hs.append(pre_h.unsqueeze(1).clone())

            pre_hs = torch.cat(pre_hs, 1)

            pooled_output = self.self_attns[idx](pre_hs) #softmax on task
            pooled_output = pooled_output.sum(1) #softmax on task

        elif self.args.task_based:
            pre_hs = []
            for pre_t in order:
                with torch.no_grad():
                    gc1,gc2,gc3,gfc1,gfc2=self.mask(t=pre_t,s=smax)
                    pre_h = \
                        self.get_feature(x,gc1,gc2,gc3,gfc1,gfc2)
                pre_hs.append(pre_h.unsqueeze(-1).clone())

            pre_hs = torch.cat(pre_hs, -1)
            pre_hs = torch.cat([pre_hs,pooled_output.unsqueeze(-1).clone()], -1) # include itselves

            pooled_output = self.self_attns[t][idx](pre_hs) #softmax on task
            pooled_output = pooled_output.sum(-1) #softmax on task

        return pooled_output




    def get_feature(self,x,gc1,gc2,gc3,gfc1,gfc2):
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



    def get_feature_augment(self,x,x_b,gc1,gc2,gc3,gfc1,gfc2,l):
        h=self.maxpool(self.drop1(self.relu(self.c1(x))))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop1(self.relu(self.c2(h))))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop2(self.relu(self.c3(h))))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=h.view(x.size(0),-1)

        h_b=self.maxpool(self.drop1(self.relu(self.c1(x_b))))
        h_b=h_b*gc1.view(1,-1,1,1).expand_as(h_b)
        h_b=self.maxpool(self.drop1(self.relu(self.c2(h_b))))
        h_b=h_b*gc2.view(1,-1,1,1).expand_as(h_b)
        h_b=self.maxpool(self.drop2(self.relu(self.c3(h_b))))
        h_b=h_b*gc3.view(1,-1,1,1).expand_as(h_b)
        h_b=h_b.view(x.size(0),-1)

        h = l*h + (1-l)*h_b


        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
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





class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,attn_size):
        super(Self_Attn,self).__init__()

        self.query_conv = nn.Linear(attn_size,attn_size)
        self.key_conv = nn.Linear(attn_size , attn_size)
        self.value_conv = nn.Linear(attn_size ,attn_size)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B,max_length,hidden_size)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        # print('x: ',x.size())
        m_batchsize,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,width,height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,width,height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # print('energy: ',energy.size())

        attention = self.softmax(energy) # BX (N) X (N)

        # attention =  F.gumbel_softmax(energy,hard=True,dim=-1)
        # print('attention: ',attention)
        proj_value = self.value_conv(x).view(m_batchsize,width,height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,width,height)

        out = self.gamma*out + x


        return out
