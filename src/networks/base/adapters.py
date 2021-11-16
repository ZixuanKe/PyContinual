from torch import nn
import torch
from config import set_args
import math
import torch.nn.functional as F
import numpy as np
import random
import sys
sys.path.append("./networks/base/")
from bayes_layer import BayesianLinear, BayesianConv2D


class BertAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1=torch.nn.Linear(config.bert_hidden_size,config.bert_adapter_size)
        self.fc2=torch.nn.Linear(config.bert_adapter_size,config.bert_hidden_size)
        if  config.use_gelu: self.activation = torch.nn.GELU()
        else: self.activation = torch.nn.ReLU()
        print('BertAdapter')

    def forward(self,x):

        h=self.activation(self.fc1(x))
        h=self.activation(self.fc2(h))

        return x + h
        # return h

    def squash(self, input_tensor, dim=-1,epsilon=1e-16): # 0 will happen in our case, has to add an epsilon
        squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
        squared_norm = squared_norm + epsilon
        scale = squared_norm / (1 + squared_norm)
        return scale * input_tensor / torch.sqrt(squared_norm)


class BertAdapterMask(BertAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.efc1=torch.nn.Embedding(config.ntasks,config.bert_adapter_size)
        self.efc2=torch.nn.Embedding(config.ntasks,config.bert_hidden_size)
        self.gate=torch.nn.Sigmoid()
        self.config = config

        print('BertAdapterMask')


    def forward(self,x,t,s):

        gfc1,gfc2=self.mask(t=t,s=s)
        h = self.get_feature(gfc1,gfc2,x)

        return x + h


    def get_feature(self,gfc1,gfc2,x):
        h=self.activation(self.fc1(x))
        h=h*gfc1.expand_as(h)

        h=self.activation(self.fc2(h))
        h=h*gfc2.expand_as(h)

        return h
    def mask(self,t,s=1):

       efc1 = self.efc1(torch.LongTensor([t]).cuda())
       efc2 = self.efc2(torch.LongTensor([t]).cuda())

       gfc1=self.gate(s*efc1)
       gfc2=self.gate(s*efc2)

       return [gfc1,gfc2]


    def my_softmax(self,input, dim=1):
        transposed_input = input.transpose(dim, len(input.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


# --------------------------
# Belows are for  CTR
# --------------------------

class BertAdapterCapsuleMaskImp(BertAdapterMask):
    '''
    Transfer Routing added
    '''

    def __init__(self, config):
        super().__init__(config)

        self.capsule_net = CapsNetImp(config)
        # self.fc1=torch.nn.Linear(semantic_cap_size,adapter_size)
        self.gelu = torch.nn.GELU()
        self.config = config
        print('BertAdapterCapsuleMaskImp')


    def forward(self,x,t,s):
        # task shared
        capsule_output = self.capsule_net(t,x,s)


        h = x + capsule_output #skip-connection

        # task specifc
        gfc1,gfc2=self.mask(t=t,s=s)

        if self.config.use_gelu:
            h=self.gelu(self.fc1(h))
            h=h*gfc1.expand_as(h)

            h=self.gelu(self.fc2(h))
            h=h*gfc2.expand_as(h)

        else:
            h=self.activation(self.fc1(h))
            h=h*gfc1.expand_as(h)

            h=self.activation(self.fc2(h))
            h=h*gfc2.expand_as(h)

        return {'outputs':x + h}


class CapsNetImp(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.semantic_capsules = CapsuleLayerImp(config,'semantic')
        if config.transfer_route:
            self.transfer_capsules = CapsuleLayerImp(config,'transfer_route')
        else:
            self.transfer_capsules = CapsuleLayerImp(config,'transfer')

        self.tsv_capsules = CapsuleLayerImp(config,'tsv')
        self.config = config
        print('CapsNet')

    def forward(self, t, x,s):
        semantic_output = self.semantic_capsules(t,x,s,'semantic')
        if self.config.transfer_route:
            transfer_output = self.transfer_capsules(t,semantic_output,s,'transfer_route')
        else:
            transfer_output = self.transfer_capsules(t,semantic_output,s,'transfer')

        tsv_output = self.tsv_capsules(t,transfer_output,s,'tsv')
        return tsv_output

class CapsuleLayerImp(nn.Module): #it has its own number of capsule for output
    def __init__(self, config,layer_type):
        super().__init__()

        if layer_type=='semantic':
            if  config.apply_one_layer_shared:
                print('apply_one_layer_shared ')
                self.fc1 = nn.ModuleList([torch.nn.Linear(config.bert_hidden_size, config.semantic_cap_size) for _ in range(config.ntasks)])

            elif config.apply_two_layer_shared:
                print('apply_two_layer_shared ')
                self.fc1 = nn.ModuleList([torch.nn.Linear(config.bert_hidden_size,config.mid_size) for _ in range(config.ntasks)])
                self.fc2 = nn.ModuleList([torch.nn.Linear(config.mid_size, config.semantic_cap_size) for _ in range(config.ntasks)])
            print('CapsuleLayer')
            self.gelu=torch.nn.GELU()

        elif layer_type=='transfer':
            D = config.semantic_cap_size
            self.Co = config.max_seq_length

            if config.semantic_cap_size == 2:
                Ks = [3,4]
            elif config.semantic_cap_size == 3:
                Ks = [3,4,5]
            elif config.semantic_cap_size == 4:
                Ks = [2,3,4,5]

            self.len_ks = len(Ks)
            self.convs1 = nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks])
            self.convs2 = nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks])
            self.convs3 = nn.ModuleList([nn.Conv1d(D, self.Co, K, padding=K-2) for K in Ks])
            self.fc_aspect = nn.Linear(self.Co*self.len_ks, self.Co)

        elif layer_type=='tsv':
            self.num_routes = config.ntasks
            self.num_capsules = config.semantic_cap_size
            self.class_dim = config.max_seq_length
            self.in_channel = config.max_seq_length*config.semantic_cap_size
            # self.in_channel = 100
            self.gate=torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax()
            self.num_iterations = 3

            #no routing for max_seq_length
            self.route_weights = \
                nn.Parameter(torch.randn(config.num_semantic_cap, self.num_routes, config.semantic_cap_size, config.semantic_cap_size))

            if config.exp in ['2layer_whole','2layer_aspect_transfer']:
                self.elarger=torch.nn.Embedding(config.ntasks,config.bert_hidden_size)
                self.larger=torch.nn.Linear(3*config.ntasks,config.bert_hidden_size) #each task has its own larger way
            else:
                self.elarger=torch.nn.Embedding(config.ntasks,config.bert_hidden_size)
                self.larger=torch.nn.Linear(config.semantic_cap_size*config.num_semantic_cap,config.bert_hidden_size) #each task has its own larger way

            if  config.no_tsv_mask:
                self.tsv = torch.ones( config.ntasks,config.ntasks).data.cuda()# for backward
            else:
                self.tsv = torch.tril(torch.ones(config.ntasks,config.ntasks)).data.cuda()# for backward



        elif layer_type=='transfer_route':

            D = config.semantic_cap_size*config.num_semantic_cap
            self.Co = 100

            Ks = [3,4,5]

            self.len_ks = len(Ks)

            # self.convs1 = nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks])
            self.convs2 = nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks])
            self.convs3 = nn.ModuleList([nn.Conv1d(D, self.Co, K, padding=K-2) for K in Ks])
            self.fc_cur = nn.Linear(self.Co*self.len_ks, self.Co)
            self.fc_sim = nn.Linear(300, config.num_semantic_cap*config.semantic_cap_size)
            self.convs4 =  nn.Conv1d(300, 2, 1)

            self.num_routes = config.ntasks
            self.num_capsules = config.semantic_cap_size
            self.class_dim = config.max_seq_length
            self.in_channel = config.max_seq_length*config.semantic_cap_size
            # self.in_channel = 100
            self.gate=torch.nn.Sigmoid()

            #no routing for max_seq_length
            self.route_weights = \
                nn.Parameter(torch.randn(config.num_semantic_cap, self.num_routes, config.semantic_cap_size, config.semantic_cap_size))

            if config.larger_as_list:
                self.larger=nn.ModuleList([
                    torch.nn.Linear(
                        config.semantic_cap_size*config.num_semantic_cap,
                        config.bert_hidden_size) for _ in range(config.ntasks)]) #each task has its own larger way
                self.elarger=torch.nn.Embedding(config.ntasks,config.bert_hidden_size) # for capusle only

            elif config.larger_as_share:
                self.larger=torch.nn.Linear(config.semantic_cap_size*config.num_semantic_cap,config.bert_hidden_size) #each task has its own larger way
                self.elarger=torch.nn.Embedding(config.ntasks,config.bert_hidden_size)

            else:
                self.elarger=torch.nn.Embedding(config.ntasks,config.bert_hidden_size)
                self.larger=torch.nn.Linear(config.semantic_cap_size*config.num_semantic_cap,config.bert_hidden_size) #each task has its own larger way



            if  config.no_tsv_mask:
                self.tsv = torch.ones( config.ntasks,config.ntasks).data.cuda()# for backward
            else:
                self.tsv = torch.tril(torch.ones(config.ntasks,config.ntasks)).data.cuda()# for backward

        self.config = config

    def forward(self, t,x,s,layer_type=None):

        if layer_type=='semantic':

            if self.config.apply_one_layer_shared:
                outputs = [fc1(x).view(x.size(0), -1, 1) for fc1 in self.fc1]
            elif self.config.apply_two_layer_shared:
                outputs = [fc2(fc1(x)).view(x.size(0), -1, 1) for fc1,fc2 in zip(self.fc1,self.fc2)]
            outputs = torch.cat(outputs, dim=-1) #(B,cap_size,19)

            outputs = self.squash(outputs)

            return outputs

        elif layer_type=='transfer_route':
                batch_size = x.size(0)
                x = x.contiguous().view(batch_size*self.config.max_seq_length,-1,self.config.semantic_cap_size)

                priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

                outputs_list = list(torch.unbind(priors,dim=2))

                decision_maker = []
                sim_attn = []
                for pre_t in range(self.config.ntasks):

                    #Regarding ASC, for numberical stable, I change relu to gelu
                        cur_v = outputs_list[t] #current_task
                        cur_v= cur_v.contiguous().view(batch_size,self.config.max_seq_length,self.config.num_semantic_cap*self.config.semantic_cap_size)

                        aa = [F.relu(conv(cur_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
                        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
                        cur_v = torch.cat(aa, 1)

                        feature = outputs_list[pre_t]
                        feature= feature.contiguous().view(batch_size,self.config.max_seq_length,self.config.num_semantic_cap*self.config.semantic_cap_size)
                        # z = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
                        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_cur(cur_v).unsqueeze(2)) for conv in self.convs2] #mix information
                        # z = [i*j for i, j in zip(z, y)]
                        # z = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in z]  # [(N,Co), ...]*len(Ks)
                        z = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]  # [(N,Co), ...]*len(Ks)

                        z = torch.cat(z, 1)
                        z= z.view(batch_size,self.Co*self.len_ks,1)
                        sim_attn.append(self.fc_sim(z.squeeze(-1)).unsqueeze(-1))
                        decision_learner = self.convs4(z).squeeze(-1)

                        gumbel_one_hot =  F.gumbel_softmax(decision_learner,hard=True)
                        # _,score = gumbel_one_hot.max(1) #hard attention gate, but not differciable
                        score = gumbel_one_hot[:,0] #hard attention gate, so that differeciable
                        decision_maker.append(score.view(-1,1))

                decision_maker = torch.cat(decision_maker, 1)
                sim_attn = torch.cat(sim_attn, 2) #TODO: Normalized the similarity

                vote_outputs = (self.tsv[t].data.view(1,1,-1,1,1) *
                    sim_attn.repeat(self.config.max_seq_length,1,1)
                                .view(self.config.num_semantic_cap,-1,self.config.ntasks,1,self.config.semantic_cap_size) *
                    decision_maker.repeat(self.config.max_seq_length,1)
                                .view(1,-1,self.config.ntasks,1,1) *
                    priors).sum(dim=2, keepdim=True) #route


                h_output = vote_outputs.view(batch_size,self.config.max_seq_length,-1)
                # print('h_output: ',h_output.size())
                if self.config.larger_as_list:
                    h_output = self.larger[t](h_output)
                elif self.config.larger_as_share:
                    h_output = self.larger(h_output)
                else:
                    h_output= self.larger(h_output)
                    glarger=self.mask(t=t,s=s)
                    h_output=h_output*glarger.expand_as(h_output)

                return h_output


        elif layer_type=='transfer':
            batch_size = x.size(0)

            if self.config.exp in ['2layer_whole']: #task,transfer,representation

                outputs_list = list(torch.unbind(x,dim=-1))
                aspect_v = outputs_list[t] #current_task
                aspect_v= aspect_v.view(batch_size,self.config.max_seq_length,self.config.semantic_cap_size)

                aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
                aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
                aspect_v = torch.cat(aa, 1)

                for pre_t in range(self.config.ntasks):
                    if pre_t != t:
                        feature = outputs_list[pre_t]
                        feature= feature.view(batch_size,self.config.max_seq_length,self.config.semantic_cap_size)
                        z = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
                        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
                        z = [i*j for i, j in zip(z, y)]
                        z = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in z]  # [(N,Co), ...]*len(Ks)
                        z = torch.cat(z, 1)
                        z= z.view(batch_size,self.Co*self.len_ks,1)
                        outputs_list[pre_t] = z
                aspect_v= aspect_v.contiguous().view(batch_size,self.Co*self.len_ks,1)
                outputs_list[t] = aspect_v

                outputs = torch.cat(outputs_list, dim=-1)
                # print('outputs: ',outputs.size())


            return outputs.transpose(2,1)



        elif layer_type=='tsv':

            if self.config.transfer_route: return x

            if self.config.exp in ['2layer_whole','2layer_aspect_transfer']: #task,transfer,representation
                batch_size = x.size(0)
                x = x.transpose(2,1)
                h_output = x.contiguous().view(batch_size,self.config.max_seq_length,-1)
                h_output= self.larger(h_output)
                glarger=self.mask(t=t,s=s)
                h_output=h_output*glarger.expand_as(h_output)
                return h_output


    def mask(self,t,s):
        glarger=self.gate(s*self.elarger(torch.LongTensor([t]).cuda()))
        return glarger

    def my_softmax(self,input, dim=1):
        transposed_input = input.transpose(dim, len(input.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

    # def squash(self, tensor, dim=-1):
    #     squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    #     scale = squared_norm / (1 + squared_norm)
    #     return scale * tensor / torch.sqrt(squared_norm)
    def squash(self, input_tensor, dim=-1,epsilon=1e-16): # 0 will happen in our case, has to add an epsilon
        squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
        squared_norm = squared_norm + epsilon
        scale = squared_norm / (1 + squared_norm)
        return scale * input_tensor / torch.sqrt(squared_norm)

# --------------------------
# Belows are for  B-CL
# --------------------------


class BertAdapterCapsuleMask(BertAdapterMask):
    def __init__(self, config):
        super().__init__(config)

        self.capsule_net = CapsNet(config)
        # self.fc1=torch.nn.Linear(semantic_cap_size,adapter_size)
        self.config = config
        print('BertAdapterCapsuleMask')


    def forward(self,x,t,s):
        # task shared
        capsule_output = self.capsule_net(t,x,s)

        h = x + capsule_output #skip-connection
        # h = capsule_output #skip-connection

        # task specifc
        gfc1,gfc2=self.mask(t=t,s=s)

        if self.config.use_gelu:
            h=self.gelu(self.fc1(h))
            h=h*gfc1.expand_as(h)

            h=self.gelu(self.fc2(h))
            h=h*gfc2.expand_as(h)
        else:
            h=self.activation(self.fc1(h))
            h=h*gfc1.expand_as(h)

            h=self.activation(self.fc2(h))
            h=h*gfc2.expand_as(h)

        return {'outputs':x + h}




class CapsNet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.semantic_capsules = CapsuleLayer(config,'semantic')
        self.tsv_capsules = CapsuleLayer(config,'tsv')
        print('CapsNet')

    def forward(self, t, x,s):
        semantic_output = self.semantic_capsules(t,x,s,'semantic')
        tsv_output = self.tsv_capsules(t,semantic_output,s,'tsv')
        return tsv_output



class CapsuleLayer(nn.Module): #it has its own number of capsule for output
    def __init__(self, config,layer_type):
        super().__init__()

        if layer_type=='tsv':
            self.num_routes = config.ntasks
            self.num_capsules = config.semantic_cap_size
            self.class_dim = config.max_seq_length
            self.in_channel = config.max_seq_length*config.semantic_cap_size
            # self.in_channel = 100
            self.elarger=torch.nn.Embedding(config.ntasks,config.bert_hidden_size)
            self.larger=torch.nn.Linear(config.semantic_cap_size,config.bert_hidden_size) #each task has its own larger way
            self.gate=torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax()
            self.num_iterations = 3
            self.route_weights = \
                nn.Parameter(torch.randn(self.num_capsules, self.num_routes, self.in_channel, self.class_dim))

            if  config.no_tsv_mask:
                self.tsv = torch.ones( config.ntasks,config.ntasks).data.cuda()# for backward
            else:
                self.tsv = torch.tril(torch.ones(config.ntasks,config.ntasks)).data.cuda()# for backward


        elif layer_type=='semantic':
            if  config.apply_one_layer_shared:
                print('apply_one_layer_shared ')
                self.fc1 = nn.ModuleList([torch.nn.Linear(config.bert_hidden_size, config.semantic_cap_size) for _ in range(config.ntasks)])

            elif config.apply_two_layer_shared:
                print('apply_two_layer_shared ')
                self.fc1 = nn.ModuleList([torch.nn.Linear(config.bert_hidden_size,100) for _ in range(config.ntasks)])
                self.fc2 = nn.ModuleList([torch.nn.Linear(100, config.semantic_cap_size) for _ in range(config.ntasks)])
            print('CapsuleLayer')

        self.config = config

    def forward(self, t,x,s,layer_type=None):

        if layer_type=='tsv':
            batch_size = x.size(0)
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = torch.zeros(*priors.size()).cuda()
            # print('logits: ',logits.size())  torch.Size([3, 32, 19, 1, 128])
            mask=torch.zeros(self.config.ntasks).data.cuda()
            # print('self.tsv[t]: ',self.tsv[t])
            for x_id in range(self.config.ntasks):
                if self.tsv[t][x_id] == 0: mask[x_id].fill_(-10000) # block future, all previous are the same

            for i in range(self.num_iterations):
                logits = logits*self.tsv[t].data.view(1,1,-1,1,1) #multiply 0 to future task
                logits = logits + mask.data.view(1,1,-1,1,1) #add a very small negative number
                probs = self.my_softmax(logits, dim=2)
                vote_outputs = (probs * priors).sum(dim=2, keepdim=True) #voted
                outputs = self.squash(vote_outputs)

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits

            h_output = vote_outputs.view(batch_size,self.config.max_seq_length,-1)

            h_output= self.larger(h_output)
            glarger=self.mask(t=t,s=s)
            h_output=h_output*glarger.expand_as(h_output)

            return h_output

        elif layer_type=='semantic':

            if self.config.apply_one_layer_shared:
                # print('apply_one_layer_shared ')
                outputs = [fc1(x).view(x.size(0), -1, 1) for fc1 in self.fc1]

            elif self.config.apply_two_layer_shared:
                # print('apply_two_layer_shared ')
                outputs = [fc2(fc1(x)).view(x.size(0), -1, 1) for fc1,fc2 in zip(self.fc1,self.fc2)]
            outputs = torch.cat(outputs, dim=-1)

            outputs = self.squash(outputs)
            return outputs.transpose(2,1)
    #
    def mask(self,t,s):
        glarger=self.gate(s*self.elarger(torch.LongTensor([t]).cuda()))
        return glarger

    def my_softmax(self,input, dim=1):
        transposed_input = input.transpose(dim, len(input.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)



# UCL MLP ==========

class BertAdapterUcl(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.fc1=BayesianLinear(config.bert_hidden_size, config.bert_adapter_size,ratio=config.ratio)
        self.fc2 = BayesianLinear(config.bert_adapter_size, config.bert_hidden_size,ratio=config.ratio)
        # self.fc1=torch.nn.Linear(config.bert_hidden_size,config.adapter_size)
        # self.fc2=torch.nn.Linear(config.adapter_size,config.bert_hidden_size)
        print('BertAdapterUcl')

    # def forward(self,x):
    #     h=self.relu(self.fc1(x))
    #     h=self.relu(self.fc2(h))
    #
    #     return x + h

    def forward(self, x, sample=False):
        if self.training: sample=True
        h = self.relu(self.fc1(x, sample))
        h = self.relu(self.fc2(h, sample))

        return x+h

# OWM MLP ==========
class BertAdapterOwm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.fc4=torch.nn.Linear(config.bert_hidden_size*config.max_seq_length,config.bert_adapter_size)
        #TODO: self.fc4=torch.nn.Linear(config.bert_hidden_size,config.adapter_size)
        self.fc1=torch.nn.Linear(config.bert_adapter_size,config.bert_adapter_size)
        self.fc2=torch.nn.Linear(config.bert_adapter_size,config.bert_adapter_size)
        self.fc3=torch.nn.Linear(config.bert_adapter_size,config.max_seq_length*config.bert_hidden_size)
        self.dropout=torch.nn.Dropout(0.5)
        self.config = config
        print('BertAdapterOwm')

    def forward(self, x, sample=False):
        h_list = []
        x_list = []
        x = x.view(x.size(0),-1)
        h = self.relu(self.fc4(x)) #OOM has to have somehting else
        h = self.relu(self.fc1(h))
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc3(h))
        h = h.view(x.size(0),self.config.max_seq_length,self.config.bert_hidden_size)
        x = x.view(x.size(0),self.config.max_seq_length,self.config.bert_hidden_size)

        return {'outputs':x + h,'x_list':x_list,'h_list':h_list}



class BertAdapterCapsule(BertAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.capsule_net = CapsNet(config)
        print('BertAdapterCapsule')

    def forward(self,x,t,s):
        h = self.capsule_net(t,x,s)
        return x + h



class BertAdapterCapsuleImp(BertAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.capsule_net = CapsNetImp(config)
        self.gelu = torch.nn.GELU()
        print('BertAdapterCapsuleImp')

    def forward(self,x,t,s):
        h = self.capsule_net(t,x,s)
        return x + h

