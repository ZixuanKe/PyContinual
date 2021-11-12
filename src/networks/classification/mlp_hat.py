import sys
import torch

import utils
import torch.nn.functional as F
import torch.nn as nn
import random
#adapter from https://github.com/joansj/hat/blob/master/src/networks/mlp_hat.py

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):
        super(Net,self).__init__()

        ncha = args.image_channel
        size = args.image_size
        nhid = args.mlp_adapter_size

        self.taskcla=taskcla
        self.args=args

        pdrop1=0.2
        pdrop2=0.5

        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)

        self.fc1=torch.nn.Linear(ncha*size*size,nhid)
        self.fc2=torch.nn.Linear(nhid,nhid)

        self.efc1=torch.nn.Embedding(len(self.taskcla),nhid)
        self.efc2=torch.nn.Embedding(len(self.taskcla),nhid)
        self.gate=torch.nn.Sigmoid()


        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(nhid,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(nhid,n))

        if 'Attn-HCHP-Outside' in self.args.mix_type:
            if self.args.attn_type=='self':
                if self.args.feature_based:
                    self.self_attns = nn.ModuleList()
                    for n in range(args.naug):
                        self.self_attns.append(Self_Attn(nhid))
                elif self.args.task_based:
                    self.self_attns = nn.ModuleList()
                    for t in range(args.ntasks):
                        self.self_attns_ = nn.ModuleList()
                        offset = 0
                        for n in range(args.naug):
                            if n > 1: offset+=1
                            if t+1-offset==0: break
                            self.self_attns_.append(Self_Attn(t+1-offset))
                        self.self_attns.append(self.self_attns_)

            elif self.args.attn_type=='cos':
                self.cos = nn.CosineSimilarity(dim=1)

                self.fc_zoomin = nn.ModuleList()
                self.fc_zoomout = nn.ModuleList()
                # self.convs4 = nn.ModuleList()

                n_orders = list(range(args.ntasks))
                self.n_order = [n_orders[:]]
                self.fc_zoomin.append(nn.Linear(nhid, args.semantic_cap_size))
                self.fc_zoomout.append(nn.Linear(args.semantic_cap_size,nhid))

                seed = args.data_seed
                for n in range(args.naug):
                    self.fc_zoomin.append(nn.Linear(nhid, args.semantic_cap_size))
                    self.fc_zoomout.append(nn.Linear(args.semantic_cap_size,nhid))
                    # random.Random(seed).shuffle(n_orders)
                    if args.nsamples > 1:
                        #you may want to only sample some: random.sample(population, k, *, counts=None)
                        n_orders_samples = random.Random(seed).sample(n_orders,args.nsamples)
                        self.n_order.append(n_orders_samples[:]) #deep copy
                    else:
                        self.n_order.append(n_orders[:]) #deep copy

                    seed+=1
                print('self.n_order: ',self.n_order)



        return

    def forward(self,t,x,start_mixup=None,s=None,l=None,idx=None,mix_type=None):
        output_dict = {}

        if start_mixup and  'Attn-HCHP-Outside' in mix_type:
            # print('attn type: ', self.args.attn_type)

            masks=self.mask(t=t,s=s)
            gfc1,gfc2=masks
            h=self.get_feature(x,gfc1,gfc2)
            if self.args.attn_type == 'self':
                h = self.self_attention_feature(t,x,h,l,idx,self.args.smax)


        else:
            # print('others: ')

            masks=self.mask(t,s=s)
            gfc1,gfc2=masks
            h=self.get_feature(x,gfc1,gfc2)

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


    def get_feature_augment(self,x,x_b,gfc1,gfc2,l):
        h=self.drop1(x.view(x.size(0),-1))
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)

        h_b=self.drop1(x_b.view(x_b.size(0),-1))
        h_b=self.drop2(self.relu(self.fc1(h_b)))
        h_b=h_b*gfc1.expand_as(h_b)
        h_b=self.drop2(self.relu(self.fc2(h_b)))
        h_b=h_b*gfc2.expand_as(h_b)


        h = l*h + (1-l)*h_b

        return h


    def self_attention_feature(self,t,x,pooled_output,order,idx,smax):
        if self.args.feature_based:
            pre_hs = []
            for pre_t in order:
                with torch.no_grad():
                    gfc1,gfc2=self.mask(t=pre_t,s=smax)
                    pre_h = \
                        self.get_feature(x,gfc1,gfc2)
                pre_hs.append(pre_h.unsqueeze(1).clone())

            pre_hs = torch.cat(pre_hs, 1)

            pooled_output = self.self_attns[idx](pre_hs) #softmax on task
            pooled_output = pooled_output.sum(1) #softmax on task

        elif self.args.task_based:
            pre_hs = []
            for pre_t in order:
                with torch.no_grad():
                    gfc1,gfc2=self.mask(t=pre_t,s=smax)
                    pre_h = \
                        self.get_feature(x,gfc1,gfc2)
                pre_hs.append(pre_h.unsqueeze(-1).clone())

            pre_hs = torch.cat(pre_hs, -1)
            pre_hs = torch.cat([pre_hs,pooled_output.unsqueeze(-1).clone()], -1) # include itselves

            pooled_output = self.self_attns[t][idx](pre_hs) #softmax on task
            pooled_output = pooled_output.sum(-1) #softmax on task

        return pooled_output


    def cos_attention_feature(self,t,x,h,l,smax):
        h = self.fc_zoomin[l](h)

        pre_hs = []
        decision_maker = []
        for pre_t in self.n_order[l]:
            with torch.no_grad():
                gfc1,gfc2=self.mask(t=pre_t,s=smax)
                pre_h = \
                    self.get_feature(x,gfc1,gfc2)
            pre_h = self.fc_zoomin[l](pre_h) # zoom in should be trainable
            pre_hs.append(pre_h.unsqueeze(1).clone())

            z = self.cos(pre_h,h) #similarity
            decision_maker.append(z.view(-1,1).clone())


        decision_maker = torch.cat(decision_maker, 1)
        pre_pooled_outputs = torch.cat(pre_hs, 1)
        pooled_output = (pre_pooled_outputs * decision_maker.unsqueeze(-1)).sum(1)
        pooled_output = self.fc_zoomout[l](pooled_output)


        return pooled_output

    def get_feature(self,x,gfc1,gfc2):
        h=self.drop1(x.view(x.size(0),-1))
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        return h


    def mask(self,t,s=1):
        gfc1=self.gate(s*self.efc1(torch.LongTensor([t]).cuda()))
        gfc2=self.gate(s*self.efc2(torch.LongTensor([t]).cuda()))
        return [gfc1,gfc2]

    def get_view_for(self,n,masks):

        gfc1,gfc2=masks

        if n=='fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)

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
