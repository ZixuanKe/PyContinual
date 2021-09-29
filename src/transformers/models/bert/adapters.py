from torch import nn
import torch
from config import set_args
import math
import torch.nn.functional as F
from bayes_layer import BayesianLinear,_calculate_fan_in_and_fan_out, BayesianConv2D
import numpy as np
import random

transformer_args = set_args()


#TODO: all self.relu, self.gelu should be change to self.activation
#TODO: adapter_size change to config., also need to change the ./network
#TODO: is_aux in config.


class BertAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1=torch.nn.Linear(config.hidden_size,config.adapter_size)
        self.fc2=torch.nn.Linear(config.adapter_size,config.hidden_size)
        if  transformer_args.use_gelu: self.activation = torch.nn.GELU()
        else: self.activation = torch.nn.ReLU()
        print('BertAdapter')

    def forward(self,x,pre_t=None,l=None,start_mixup=False,idx=None):

        if start_mixup and pre_t=='tmix':
            h=self.activation(self.fc1(x))
            h=self.activation(self.fc2(h))

            idx_h=self.activation(self.fc1(x[idx]))
            idx_h=self.activation(self.fc2(idx_h))

            h=l * h + (1-l) * idx_h

        else:
            h=self.activation(self.fc1(x))
            h=self.activation(self.fc2(h))

        return x + h
        # return h

    def squash(self, input_tensor, dim=-1,epsilon=1e-16): # 0 will happen in our case, has to add an epsilon
        squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
        squared_norm = squared_norm + epsilon
        scale = squared_norm / (1 + squared_norm)
        return scale * input_tensor / torch.sqrt(squared_norm)



class BertAdapterGrow(BertAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.fc1=\
            torch.nn.ModuleList([torch.nn.Linear(config.hidden_size,transformer_args.adapter_size) for _ in range(transformer_args.ntasks)])
        self.fc2=\
            torch.nn.ModuleList([torch.nn.Linear(transformer_args.adapter_size,config.hidden_size) for _ in range(transformer_args.ntasks)])

        print('BertAdapterGrow')

    def forward(self,x,t):
        h=self.activation(self.fc1[t](x))
        h=self.activation(self.fc2[t](h))

        return x + h
        # return h


class BertAdapterMask(BertAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.efc1=torch.nn.Embedding(transformer_args.ntasks,config.adapter_size)
        self.efc2=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size)
        self.gate=torch.nn.Sigmoid()
        self.config = config

        print('BertAdapterMask')


    def forward(self,x,t,s,smax=400,pre_t=None,l=None,start_mixup=False,idx=None):

        if start_mixup and pre_t=='tmix':
            gfc1,gfc2=self.mask(t=t,s=smax)

            h = self.get_feature(gfc1,gfc2,x)
            idx_h = self.get_feature(gfc1,gfc2,x[idx])

            h=l * h + (1-l) * idx_h
        else:
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




class BertAdapterCapsuleMaskImp(BertAdapterMask):
    '''
    Transfer Routing added
    '''

    def __init__(self, config):
        super().__init__(config)

        self.capsule_net = CapsNetImp(config)
        # self.fc1=torch.nn.Linear(semantic_cap_size,adapter_size)
        self.gelu = torch.nn.GELU()
        print('BertAdapterCapsuleMaskImp')


    def forward(self,x,t,s,targets,token_type_ids,smax=400):
        # task shared
        margin_loss = 0
        capsule_output = self.capsule_net(t,x,s,token_type_ids)

        # if torch.isinf(capsule_output).any() or torch.isnan(capsule_output).any():
        #     print('clamp capsule_output')
        #     clamp_value = torch.finfo(capsule_output.dtype).max - 1000
        #     capsule_output = torch.clamp(capsule_output, min=-clamp_value, max=clamp_value)
        #for numerical stable ================

        h = x + capsule_output #skip-connection
        # h = capsule_output #skip-connection

        # task specifc
        gfc1,gfc2=self.mask(t=t,s=s)

        if transformer_args.use_gelu:
            h=self.gelu(self.fc1(h))
            h=h*gfc1.expand_as(h)

            h=self.gelu(self.fc2(h))
            h=h*gfc2.expand_as(h)

            # if torch.isinf(h).any() or torch.isnan(h).any():
            #     print('clamp h')
            #     clamp_value = torch.finfo(h.dtype).max - 1000
            #     h = torch.clamp(h, min=-clamp_value, max=clamp_value)
            #for numerical stable ================

        else:
            h=self.activation(self.fc1(h))
            h=h*gfc1.expand_as(h)

            h=self.activation(self.fc2(h))
            h=h*gfc2.expand_as(h)

        return {'outputs':x + h,'margin_loss':margin_loss}


class CapsNetImp(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.semantic_capsules = CapsuleLayerImp(config,'semantic')
        if transformer_args.transfer_route:
            self.transfer_capsules = CapsuleLayerImp(config,'transfer_route')
        else:
            self.transfer_capsules = CapsuleLayerImp(config,'transfer')

        self.tsv_capsules = CapsuleLayerImp(config,'tsv')
        print('CapsNet')

    def forward(self, t, x,s,token_type_ids=None):
        semantic_output = self.semantic_capsules(t,x,s,token_type_ids,'semantic')
        if transformer_args.transfer_route:
            transfer_output = self.transfer_capsules(t,semantic_output,s,token_type_ids,'transfer_route')
        else:
            transfer_output = self.transfer_capsules(t,semantic_output,s,token_type_ids,'transfer')

        tsv_output = self.tsv_capsules(t,transfer_output,s,token_type_ids,'tsv')
        return tsv_output

class CapsuleLayerImp(nn.Module): #it has its own number of capsule for output
    def __init__(self, config,layer_type):
        super().__init__()

        if layer_type=='semantic':
            if  transformer_args.apply_one_layer_shared:
                print('apply_one_layer_shared ')
                self.fc1 = nn.ModuleList([torch.nn.Linear(config.hidden_size, transformer_args.semantic_cap_size) for _ in range(transformer_args.ntasks)])

            elif transformer_args.apply_two_layer_shared:
                print('apply_two_layer_shared ')
                self.fc1 = nn.ModuleList([torch.nn.Linear(config.hidden_size,transformer_args.mid_size) for _ in range(transformer_args.ntasks)])
                self.fc2 = nn.ModuleList([torch.nn.Linear(transformer_args.mid_size, transformer_args.semantic_cap_size) for _ in range(transformer_args.ntasks)])
            print('CapsuleLayer')
            self.gelu=torch.nn.GELU()

        elif layer_type=='transfer':
            D = transformer_args.semantic_cap_size
            self.Co = transformer_args.max_seq_length
            # Ks = [3,4,5]
            # Ks = [2,3,4]

            if transformer_args.semantic_cap_size == 2:
                Ks = [3,4]
            elif transformer_args.semantic_cap_size == 3:
                Ks = [3,4,5]
            elif transformer_args.semantic_cap_size == 4:
                Ks = [2,3,4,5]

            self.len_ks = len(Ks)
            self.convs1 = nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks])
            self.convs2 = nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks])
            self.convs3 = nn.ModuleList([nn.Conv1d(D, self.Co, K, padding=K-2) for K in Ks])
            self.fc_aspect = nn.Linear(self.Co*self.len_ks, self.Co)

        elif layer_type=='tsv':
            self.num_routes = transformer_args.ntasks
            self.num_capsules = transformer_args.semantic_cap_size
            self.class_dim = transformer_args.max_seq_length
            self.in_channel = transformer_args.max_seq_length*transformer_args.semantic_cap_size
            # self.in_channel = 100
            self.gate=torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax()
            self.num_iterations = 3

            #no routing for max_seq_length
            self.route_weights = \
                nn.Parameter(torch.randn(transformer_args.num_semantic_cap, self.num_routes, transformer_args.semantic_cap_size, transformer_args.semantic_cap_size))

            if transformer_args.exp in ['2layer_whole','2layer_aspect_transfer']:
                self.elarger=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size)
                self.larger=torch.nn.Linear(3*transformer_args.ntasks,config.hidden_size) #each task has its own larger way
            else:
                self.elarger=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size)
                self.larger=torch.nn.Linear(transformer_args.semantic_cap_size*transformer_args.num_semantic_cap,config.hidden_size) #each task has its own larger way

            # else: this capsule is with max_length, much slower
            #     self.route_weights = \
            #         nn.Parameter(torch.randn(self.num_capsules, self.num_routes, self.in_channel, self.class_dim))
            #     self.elarger=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size)
            #     self.larger=torch.nn.Linear(transformer_args.semantic_cap_size,config.hidden_size) #each task has its own larger way

            if  transformer_args.no_tsv_mask:
                self.tsv = torch.ones( transformer_args.ntasks,transformer_args.ntasks).data.cuda()# for backward
            else:
                self.tsv = torch.tril(torch.ones(transformer_args.ntasks,transformer_args.ntasks)).data.cuda()# for backward



        elif layer_type=='transfer_route':

            D = transformer_args.semantic_cap_size*transformer_args.num_semantic_cap
            self.Co = 100
            # Ks = [3,4,5]
            # Ks = [2,3,4]

            Ks = [3,4,5]

            self.len_ks = len(Ks)

            # self.convs1 = nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks])
            self.convs2 = nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks])
            self.convs3 = nn.ModuleList([nn.Conv1d(D, self.Co, K, padding=K-2) for K in Ks])
            self.fc_cur = nn.Linear(self.Co*self.len_ks, self.Co)
            self.fc_sim = nn.Linear(300, transformer_args.num_semantic_cap*transformer_args.semantic_cap_size)
            self.convs4 =  nn.Conv1d(300, 2, 1)

            self.num_routes = transformer_args.ntasks
            self.num_capsules = transformer_args.semantic_cap_size
            self.class_dim = transformer_args.max_seq_length
            self.in_channel = transformer_args.max_seq_length*transformer_args.semantic_cap_size
            # self.in_channel = 100
            self.gate=torch.nn.Sigmoid()

            #no routing for max_seq_length
            self.route_weights = \
                nn.Parameter(torch.randn(transformer_args.num_semantic_cap, self.num_routes, transformer_args.semantic_cap_size, transformer_args.semantic_cap_size))

            # self.elarger=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size)
            # self.larger=torch.nn.Linear(transformer_args.semantic_cap_size*transformer_args.num_semantic_cap,config.hidden_size) #each task has its own larger way

            if transformer_args.larger_as_list:
                self.larger=nn.ModuleList([
                    torch.nn.Linear(
                        transformer_args.semantic_cap_size*transformer_args.num_semantic_cap,
                        config.hidden_size) for _ in range(transformer_args.ntasks)]) #each task has its own larger way
                self.elarger=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size) # for capusle only

            elif transformer_args.larger_as_share:
                self.larger=torch.nn.Linear(transformer_args.semantic_cap_size*transformer_args.num_semantic_cap,config.hidden_size) #each task has its own larger way
                self.elarger=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size)

            else:
                self.elarger=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size)
                self.larger=torch.nn.Linear(transformer_args.semantic_cap_size*transformer_args.num_semantic_cap,config.hidden_size) #each task has its own larger way



            if  transformer_args.no_tsv_mask:
                self.tsv = torch.ones( transformer_args.ntasks,transformer_args.ntasks).data.cuda()# for backward
            else:
                self.tsv = torch.tril(torch.ones(transformer_args.ntasks,transformer_args.ntasks)).data.cuda()# for backward


    def forward(self, t,x,s,token_type_ids=None,layer_type=None):

        if layer_type=='semantic':

            if transformer_args.apply_one_layer_shared:
                outputs = [fc1(x).view(x.size(0), -1, 1) for fc1 in self.fc1]
            elif transformer_args.apply_two_layer_shared:
                outputs = [fc2(fc1(x)).view(x.size(0), -1, 1) for fc1,fc2 in zip(self.fc1,self.fc2)]
            outputs = torch.cat(outputs, dim=-1) #(B,cap_size,19)

            outputs = self.squash(outputs)

            #for numerical stable ================
            # if torch.isinf(outputs).any() or torch.isnan(outputs).any():
            #     print('clamp outputs')
            #     clamp_value = torch.finfo(outputs.dtype).max - 1000
            #     outputs = torch.clamp(outputs, min=-clamp_value, max=clamp_value)
            #for numerical stable ================
            return outputs

        elif layer_type=='transfer_route':
                batch_size = x.size(0)
                x = x.contiguous().view(batch_size*transformer_args.max_seq_length,-1,transformer_args.semantic_cap_size)

                priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

                outputs_list = list(torch.unbind(priors,dim=2))

                decision_maker = []
                sim_attn = []
                for pre_t in range(transformer_args.ntasks):

                    # if pre_t == t:
                    #     score = torch.LongTensor([1]).cuda().repeat(batch_size)
                    #     decision_maker.append(score.view(-1,1))
                    #
                    # elif pre_t > t:
                    #     score = torch.LongTensor([0]).cuda().repeat(batch_size)
                    #     decision_maker.append(score.view(-1,1))

                    # else:

                    #Regarding ASC, for numberical stable, I change relu to gelu
                        cur_v = outputs_list[t] #current_task
                        cur_v= cur_v.contiguous().view(batch_size,transformer_args.max_seq_length,transformer_args.num_semantic_cap*transformer_args.semantic_cap_size)

                        aa = [F.relu(conv(cur_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
                        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
                        cur_v = torch.cat(aa, 1)

                        feature = outputs_list[pre_t]
                        feature= feature.contiguous().view(batch_size,transformer_args.max_seq_length,transformer_args.num_semantic_cap*transformer_args.semantic_cap_size)
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


                #for numerical stable ================
                # mask=torch.zeros(transformer_args.ntasks).data.cuda()
                # # print('self.tsv[t]: ',self.tsv[t])
                # for x_id in range(transformer_args.ntasks):
                #     if self.tsv[t][x_id] == 0: mask[x_id].fill_(-10000) # block future, all previous are the same
                #
                # decision_maker = \
                #     decision_maker.repeat(transformer_args.max_seq_length,1).view(1,-1,transformer_args.ntasks,1,1)*\
                #     self.tsv[t].data.view(1,1,-1,1,1) #multiply 0 to future task
                #
                # decision_maker = decision_maker + mask.data.view(1,1,-1,1,1) #add a very small negative number; mask for softmax
                # probs = self.my_softmax(decision_maker, dim=2) #include decision and masking
                #
                # vote_outputs = (probs *
                #     sim_attn.repeat(transformer_args.max_seq_length,1,1)
                #                 .view(transformer_args.num_semantic_cap,-1,transformer_args.ntasks,1,transformer_args.semantic_cap_size) *
                #     priors).sum(dim=2, keepdim=True) #route
                #
                #for numerical stable ================
                # epsilon=1e-16
                #for numerical stable ================


                vote_outputs = (self.tsv[t].data.view(1,1,-1,1,1) *
                    sim_attn.repeat(transformer_args.max_seq_length,1,1)
                                .view(transformer_args.num_semantic_cap,-1,transformer_args.ntasks,1,transformer_args.semantic_cap_size) *
                    decision_maker.repeat(transformer_args.max_seq_length,1)
                                .view(1,-1,transformer_args.ntasks,1,1) *
                    priors).sum(dim=2, keepdim=True) #route

                #for numerical stable ================
                # if torch.isinf(vote_outputs).any() or torch.isnan(vote_outputs).any():
                #     print('clamp vote_outputs')
                #     clamp_value = torch.finfo(vote_outputs.dtype).max - 1000
                #     vote_outputs = torch.clamp(vote_outputs, min=-clamp_value, max=clamp_value)
                #for numerical stable ================

                h_output = vote_outputs.view(batch_size,transformer_args.max_seq_length,-1)
                # print('h_output: ',h_output.size())
                if transformer_args.larger_as_list:
                    h_output = self.larger[t](h_output)
                elif transformer_args.larger_as_share:
                    h_output = self.larger(h_output)
                else:
                    h_output= self.larger(h_output)
                    glarger=self.mask(t=t,s=s)
                    h_output=h_output*glarger.expand_as(h_output)

                return h_output



        elif layer_type=='transfer':
            batch_size = x.size(0)

            if transformer_args.exp in ['3layer_whole','2layer_whole']: #task,transfer,representation

                outputs_list = list(torch.unbind(x,dim=-1))
                aspect_v = outputs_list[t] #current_task
                aspect_v= aspect_v.view(batch_size,transformer_args.max_seq_length,transformer_args.semantic_cap_size)

                aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
                aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
                aspect_v = torch.cat(aa, 1)

                for pre_t in range(transformer_args.ntasks):
                    if pre_t != t:
                        feature = outputs_list[pre_t]
                        feature= feature.view(batch_size,transformer_args.max_seq_length,transformer_args.semantic_cap_size)
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

            elif transformer_args.exp in ['3layer_aspect','2layer_aspect_transfer']: #task,transfer,representation
                # aspect routing: we know the similar part is actually the aspect.
                # Why not mannual route by aspect.
                # instead of using CNN to learn the aspect is important and then route in next layer
                # need to make use the whole thing at the end, consider "adding" to the to sentence

                outputs_list = list(torch.unbind(x,dim=-1))
                whole_v = outputs_list[t] #current_task
                whole_v= whole_v.view(batch_size,transformer_args.max_seq_length,transformer_args.semantic_cap_size)
                aspect_v = whole_v[:,1:1+transformer_args.max_term_length,:]

                aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
                aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
                aspect_v = torch.cat(aa, 1)

                for pre_t in range(transformer_args.ntasks):
                    if pre_t != t:
                        whole_feature = outputs_list[pre_t]
                        whole_feature= whole_feature.view(batch_size,transformer_args.max_seq_length,transformer_args.semantic_cap_size)
                        feature = whole_feature[:,1:1+transformer_args.max_term_length,:]

                        z = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
                        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
                        z = [i*j for i, j in zip(z, y)]
                        z = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in z]  # [(N,Co), ...]*len(Ks)
                        z = torch.cat(z, 1)
                        z= z.view(batch_size,self.Co*self.len_ks,1)
                        outputs_list[pre_t] = z
                aspect_v= aspect_v.contiguous().view(batch_size,self.Co*self.len_ks,1)
                outputs_list[t] = aspect_v

                outputs = torch.cat(outputs_list, dim=-1) + x #we still need original sentence
                # print('outputs: ',outputs.size())

            elif transformer_args.exp == '2layer_aspect_dynamic':
                outputs = x


            return outputs.transpose(2,1)



        elif layer_type=='tsv':

            if transformer_args.transfer_route: return x

            if '3layer' in transformer_args.exp: #task,transfer,representation

                batch_size = x.size(0)
                x = x.contiguous().view(batch_size*transformer_args.max_seq_length,-1,transformer_args.semantic_cap_size)

                priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
                logits = torch.zeros(*priors.size()).cuda()
                mask=torch.zeros(transformer_args.ntasks).data.cuda()
                # print('self.tsv[t]: ',self.tsv[t])
                for x_id in range(transformer_args.ntasks):
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

                h_output = vote_outputs.view(batch_size,transformer_args.max_seq_length,-1)

                h_output= self.larger(h_output)
                glarger=self.mask(t=t,s=s)
                h_output=h_output*glarger.expand_as(h_output)

                return h_output

            elif transformer_args.exp in ['2layer_whole','2layer_aspect_transfer']: #task,transfer,representation
                batch_size = x.size(0)
                x = x.transpose(2,1)
                h_output = x.contiguous().view(batch_size,transformer_args.max_seq_length,-1)
                h_output= self.larger(h_output)
                glarger=self.mask(t=t,s=s)
                h_output=h_output*glarger.expand_as(h_output)
                return h_output

            elif transformer_args.exp == '2layer_aspect_dynamic': #task,transfer,representation

                batch_size = x.size(0)
                x = x.contiguous().view(batch_size*transformer_args.max_seq_length,-1,transformer_args.semantic_cap_size)

                priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
                logits = torch.zeros(*priors.size()).cuda()
                # print('logits: ',logits.size())  torch.Size([3, 32, 19, 1, 128])
                mask=torch.zeros(transformer_args.ntasks).data.cuda()
                # print('self.tsv[t]: ',self.tsv[t])
                for x_id in range(transformer_args.ntasks):
                    if self.tsv[t][x_id] == 0: mask[x_id].fill_(-10000) # block future, all previous are the same

                for i in range(self.num_iterations):
                    logits = logits*self.tsv[t].data.view(1,1,-1,1,1) #multiply 0 to future task
                    logits = logits + mask.data.view(1,1,-1,1,1) #add a very small negative number
                    probs = self.my_softmax(logits, dim=2)
                    vote_outputs = (probs * priors).sum(dim=2, keepdim=True) #voted
                    outputs = self.squash(vote_outputs)
                    if i != self.num_iterations - 1:
                        priors_ = \
                            priors.view(
                                transformer_args.num_semantic_cap,
                                batch_size,
                                transformer_args.max_seq_length,
                                transformer_args.ntasks,
                                1,
                                transformer_args.semantic_cap_size) [:,:,1:1+transformer_args.max_term_length,:,:,:]

                        outputs_ = \
                            outputs.view(
                                transformer_args.num_semantic_cap,
                                batch_size,
                                transformer_args.max_seq_length,
                                1,
                                1,
                                transformer_args.semantic_cap_size) [:,:,1:1+transformer_args.max_term_length,:,:,:]

                        #delta only has something to do with aspect

                        delta_logits = (priors_* outputs_).sum(dim=-1, keepdim=True).sum(dim=2, keepdim=False)
                        delta_logits = delta_logits.repeat(1,transformer_args.max_seq_length,1,1,1)
                        #only compare the aspect

                        logits = logits + delta_logits

                h_output = vote_outputs.view(batch_size,transformer_args.max_seq_length,-1)
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
#====================== Belows are OLD B-CL


class BertAdapterCapsuleMask(BertAdapterMask):
    def __init__(self, config):
        super().__init__(config)

        self.capsule_net = CapsNet(config)
        # self.fc1=torch.nn.Linear(semantic_cap_size,adapter_size)
        print('BertAdapterCapsuleMask')


    def forward(self,x,t,s,targets,token_type_ids,smax=400):
        # task shared
        margin_loss = 0
        capsule_output = self.capsule_net(t,x,s)

        h = x + capsule_output #skip-connection
        # h = capsule_output #skip-connection

        # task specifc
        gfc1,gfc2=self.mask(t=t,s=s)

        if transformer_args.use_gelu:
            h=self.gelu(self.fc1(h))
            h=h*gfc1.expand_as(h)

            h=self.gelu(self.fc2(h))
            h=h*gfc2.expand_as(h)
        else:
            h=self.activation(self.fc1(h))
            h=h*gfc1.expand_as(h)

            h=self.activation(self.fc2(h))
            h=h*gfc2.expand_as(h)

        return {'outputs':x + h,'margin_loss':margin_loss}




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
            self.num_routes = transformer_args.ntasks
            self.num_capsules = transformer_args.semantic_cap_size
            self.class_dim = transformer_args.max_seq_length
            self.in_channel = transformer_args.max_seq_length*transformer_args.semantic_cap_size
            # self.in_channel = 100
            self.elarger=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size)
            self.larger=torch.nn.Linear(transformer_args.semantic_cap_size,config.hidden_size) #each task has its own larger way
            self.gate=torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax()
            self.num_iterations = 3
            self.route_weights = \
                nn.Parameter(torch.randn(self.num_capsules, self.num_routes, self.in_channel, self.class_dim))

            if  transformer_args.no_tsv_mask:
                self.tsv = torch.ones( transformer_args.ntasks,transformer_args.ntasks).data.cuda()# for backward
            else:
                self.tsv = torch.tril(torch.ones(transformer_args.ntasks,transformer_args.ntasks)).data.cuda()# for backward


        elif layer_type=='semantic':
            if  transformer_args.apply_one_layer_shared:
                print('apply_one_layer_shared ')
                self.fc1 = nn.ModuleList([torch.nn.Linear(config.hidden_size, transformer_args.semantic_cap_size) for _ in range(transformer_args.ntasks)])

            elif transformer_args.apply_two_layer_shared:
                print('apply_two_layer_shared ')
                self.fc1 = nn.ModuleList([torch.nn.Linear(config.hidden_size,100) for _ in range(transformer_args.ntasks)])
                self.fc2 = nn.ModuleList([torch.nn.Linear(100, transformer_args.semantic_cap_size) for _ in range(transformer_args.ntasks)])
            print('CapsuleLayer')

    def forward(self, t,x,s,layer_type=None):

        if layer_type=='tsv':
            batch_size = x.size(0)
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = torch.zeros(*priors.size()).cuda()
            # print('logits: ',logits.size())  torch.Size([3, 32, 19, 1, 128])
            mask=torch.zeros(transformer_args.ntasks).data.cuda()
            # print('self.tsv[t]: ',self.tsv[t])
            for x_id in range(transformer_args.ntasks):
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

            h_output = vote_outputs.view(batch_size,transformer_args.max_seq_length,-1)

            h_output= self.larger(h_output)
            glarger=self.mask(t=t,s=s)
            h_output=h_output*glarger.expand_as(h_output)

            return h_output

        elif layer_type=='semantic':

            if transformer_args.apply_one_layer_shared:
                # print('apply_one_layer_shared ')
                outputs = [fc1(x).view(x.size(0), -1, 1) for fc1 in self.fc1]

            elif transformer_args.apply_two_layer_shared:
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
        self.fc1=BayesianLinear(config.hidden_size, config.adapter_size,ratio=transformer_args.ratio)
        self.fc2 = BayesianLinear(config.adapter_size, config.hidden_size,ratio=transformer_args.ratio)
        # self.fc1=torch.nn.Linear(config.hidden_size,transformer_args.adapter_size)
        # self.fc2=torch.nn.Linear(transformer_args.adapter_size,config.hidden_size)
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
        self.fc4=torch.nn.Linear(config.hidden_size*transformer_args.max_seq_length,config.adapter_size)
        #TODO: self.fc4=torch.nn.Linear(config.hidden_size,transformer_args.adapter_size)
        self.fc1=torch.nn.Linear(config.adapter_size,config.adapter_size)
        self.fc2=torch.nn.Linear(config.adapter_size,config.adapter_size)
        self.fc3=torch.nn.Linear(config.adapter_size,transformer_args.max_seq_length*config.hidden_size)
        self.dropout=torch.nn.Dropout(0.5)

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
        h = h.view(x.size(0),transformer_args.max_seq_length,transformer_args.bert_hidden_size)
        x = x.view(x.size(0),transformer_args.max_seq_length,transformer_args.bert_hidden_size)

        return {'outputs':x + h,'x_list':x_list,'h_list':h_list}



# ===================================== belows are chcked, but no good ==============================================

class BertAdapterMlpMask(BertAdapterMask):
    def __init__(self, config):
        super().__init__(config)
        self.capsule_net = CapsNet(config)
        self.fc3=torch.nn.Linear(config.hidden_size,transformer_args.adapter_size)
        self.fc4=torch.nn.Linear(transformer_args.adapter_size,config.hidden_size)
        print('BertAdapterMlpMask')

    def forward(self,x,t,s):
        # task shared

        h=self.activation(self.fc3(x))
        h=self.activation(self.fc4(h))

        # task specifc
        gfc1,gfc2=self.mask(t=t,s=s)

        h=self.activation(self.fc1(h))
        h=h*gfc1.expand_as(h)

        h=self.activation(self.fc2(h))
        h=h*gfc2.expand_as(h)

        return x + h

class BertAdapterCapsule(BertAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.capsule_net = CapsNet(config)
        print('BertAdapterCapsule')

    def forward(self,x,t,s,targets,smax=400):
        h = self.capsule_net(t,x,s)
        return x + h



class BertAdapterCapsuleImp(BertAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.capsule_net = CapsNetImp(config)
        self.gelu = torch.nn.GELU()
        print('BertAdapterCapsuleImp')

    def forward(self,x,t,s,targets,smax=400):
        h = self.capsule_net(t,x,s)
        return x + h


class BertAdapterTwoModules(BertAdapterMask):
    '''
    One for specific, one for shared
    '''
    def __init__(self, config):
        super().__init__(config)
        self.shared_fc1 = torch.nn.Linear(config.hidden_size,transformer_args.adapter_size)
        self.shared_fc2 = torch.nn.Linear(transformer_args.adapter_size,config.hidden_size//2)
        self.fc2=torch.nn.Linear(transformer_args.adapter_size,config.hidden_size//2)
        self.efc2=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size//2)
        self.mse_loss = torch.nn.MSELoss()
        print('BertAdapterTwoModules')


    def forward(self,x,t,s,smax=400):
        gfc1,gfc2=self.mask(t=t,s=s)

        # task-specifc
        h=self.activation(self.fc1(x))
        h=h*gfc1.expand_as(h)

        h=self.activation(self.fc2(h))
        h=h*gfc2.expand_as(h)


        # shared
        shared_h=self.activation(self.shared_fc1(x))
        shared_h=self.activation(self.shared_fc2(shared_h))

        #reconstruct needed
        recon_loss=0
        if t > 0:
            for pre_t in range(t):
                pre_gfc1,pre_gfc2=self.mask(pre_t,s=smax)
                pre_gfc1 = pre_gfc1.data.detach()
                pre_gfc2 = pre_gfc2.data.detach()

                pre_h=self.activation(self.fc1(x)).data.detach()
                pre_h=pre_h*pre_gfc1.expand_as(pre_h).data.detach()

                pre_h=self.activation(self.fc2(pre_h)).data.detach()
                pre_h=pre_h*pre_gfc2.expand_as(pre_h).data.detach()

                recon_loss +=self.mse_loss(shared_h, pre_h)

        cat_h = torch.cat([h,shared_h],-1)
        # print('recon_loss: ',recon_loss)

        return {'outputs':x + cat_h,'recon_loss':recon_loss}
        # return h



class BertAdapterAttentionMask(BertAdapterMask):
    # this is attended on different previous task representation.
    # but those representaiton can be really specific and thus attend on them are of no help

    def __init__(self, config):
        super().__init__(config)

        self.attention = BertAttentionMask(config)
        self.etask=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size)
        print('BertAdapterAttentionMask')

    def forward(self,x,t,s,smax=400):
        gfc1,gfc2=self.mask(t=t,s=s)

        h=self.activation(self.fc1(x))
        h=h*gfc1.expand_as(h)

        h=self.activation(self.fc2(h))
        h=h*gfc2.expand_as(h)

        if t > 0:
            pre_models = []
            for pre_t in range(t):

                pre_gfc1,pre_gfc2=self.mask(pre_t,s=smax)
                pre_gfc1 = pre_gfc1.data.detach()
                pre_gfc2 = pre_gfc2.data.detach()

                pre_h=self.activation(self.fc1(x)).data.detach()
                pre_h=pre_h*pre_gfc1.expand_as(pre_h).data.detach()

                pre_h=self.activation(self.fc2(pre_h)).data.detach()
                pre_h=pre_h*pre_gfc2.expand_as(pre_h).data.detach()

                pre_models.append(pre_h.data.clone().detach())

            pre_models = torch.stack(pre_models,1).data.detach() # previous models are fixed and thus detached
            h = torch.cat([h.unsqueeze(1),pre_models],1) #h itself is trainable
            t_embedding = self.etask(torch.LongTensor([t]).cuda()).repeat(h.size(0),h.size(2),1).unsqueeze(1)

            h = \
                self.attention(query_states=t_embedding,hidden_states=h,t=t,s=s)
            h = h[0]

        return x + h
        # return h



class BertAttentionMask(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttentionMask(config)
        # self.output = BertSelfOutput(config)
        #TODO: we may need output later
        print('BertAttentionMask')

    def forward(
        self,
        query_states,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,t=None,s=1

    ):
        # print('BertAttentionMask')
        self_outputs = self.self(query_states=query_states,hidden_states=hidden_states,t=t,s=s)

        # attention_output = self.output(self_outputs[0], hidden_states,t=t,s=s)
        # outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # return outputs
        return self_outputs


class BertSelfAttentionMask(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.equery=torch.nn.Embedding(transformer_args.ntasks,self.all_head_size)
        self.ekey=torch.nn.Embedding(transformer_args.ntasks,self.all_head_size)
        self.evalue=torch.nn.Embedding(transformer_args.ntasks,self.all_head_size)
        self.gate=torch.nn.Sigmoid()
        print('BertSelfAttentionMask')

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,1,3, 2, 4)



    def mask(self,t,s=1):
        # print('BertSelfAttentionMask mask')
        gquery=self.gate(s*self.equery(torch.LongTensor([t]).cuda()))
        gkey=self.gate(s*self.ekey(torch.LongTensor([t]).cuda()))
        gvalue=self.gate(s*self.evalue(torch.LongTensor([t]).cuda()))
        return [gquery,gkey,gvalue]

    def forward(
        self,
        query_states,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,t=None,s=1
    ):

        # print('BertSelfAttentionMask forward')
        gquery,gkey,gvalue=self.mask(t=t,s=s)

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        mixed_query_layer=mixed_query_layer*gquery.expand_as(mixed_query_layer)
        mixed_key_layer=mixed_key_layer*gkey.expand_as(mixed_key_layer)
        mixed_value_layer=mixed_value_layer*gvalue.expand_as(mixed_value_layer)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        #batch matrix multiplication
        #order is also matter


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = \
            torch.matmul(query_layer.permute(0,4,3,1,2), key_layer.permute(0,4,3,2,1))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)



        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer.permute(0,4,3,1,2))

        #2 -> 1 and can be squeezed

        context_layer = context_layer.squeeze(-2).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs




class BertAdapterCapsuleMaskImpAllMask(BertAdapterMask):
    '''
    Use task masking to generate different task representation
    '''
    def __init__(self, config):
        super().__init__(config)

        self.num_iterations = 3

        self.gelu = torch.nn.GELU()

        #****************************************
        #Task capsule
        #****************************************

        self.fc1=torch.nn.Linear(config.hidden_size,transformer_args.adapter_size)
        self.fc2=torch.nn.Linear(transformer_args.adapter_size,transformer_args.mid_size)
        self.efc1=torch.nn.Embedding(transformer_args.ntasks,transformer_args.adapter_size)
        self.efc2=torch.nn.Embedding(transformer_args.ntasks,transformer_args.mid_size)
        #****************************************
        #Transfer capsule
        #****************************************

        # D = transformer_args.bert_hidden_size
        # self.Co = transformer_args.cnn_kernel_size
        # Ks = [3,4,5]
        #
        # self.convs1 = nn.ModuleList([nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks]) for _ in range(transformer_args.ntasks)])
        # self.convs2 = nn.ModuleList([nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks]) for _ in range(transformer_args.ntasks)])
        # self.convs3 = nn.ModuleList([nn.ModuleList([nn.Conv1d(D, self.Co, K, padding=K-2) for K in Ks]) for _ in range(transformer_args.ntasks)])
        # self.fc_aspect = nn.ModuleList([nn.Linear(self.Co*3, self.Co) for _ in range(transformer_args.ntasks)])

        #****************************************
        #Representation capsule
        #****************************************

        self.elarger=torch.nn.Embedding(transformer_args.ntasks,config.hidden_size)
        self.larger=torch.nn.Linear(transformer_args.semantic_cap_size,config.hidden_size) #each task has its own larger way


        self.route_weights = \
            nn.Parameter(torch.randn(transformer_args.semantic_cap_size,
                                     transformer_args.ntasks,
                                     transformer_args.max_seq_length*transformer_args.mid_size,
                                     transformer_args.max_seq_length))

        if  transformer_args.no_tsv_mask:
            self.tsv = torch.ones( transformer_args.ntasks,transformer_args.ntasks).data.cuda()# for backward
        else:
            self.tsv = torch.tril(torch.ones(transformer_args.ntasks,transformer_args.ntasks)).data.cuda()# for backward

        print('BertAdapterCapsuleMaskImp')

    def forward(self,x,t,s,targets):
        #****************************************
        #task capsule
        #****************************************
        outputs = []
        batch_size = x.size(0)
        for pre_t in range(transformer_args.ntasks):

            if pre_t <= t:
                masks=self.mask(pre_t,s=transformer_args.smax)
                gfc1,gfc2,glarger=masks

                h=self.gelu(self.fc1(x))
                h=h*gfc1.expand_as(h)

                h=self.gelu(self.fc2(h))
                h=h*gfc2.expand_as(h)
                outputs.append(h.view(x.size(0), -1, 1))

            else:
                pre_masks=self.mask(pre_t,s=transformer_args.smax)
                pre_gfc1,pre_gfc2,_=pre_masks

                pre_h=self.gelu(self.fc1(x))
                pre_h=pre_h*pre_gfc1.data.clone().expand_as(pre_h)

                pre_h=self.gelu(self.fc2(pre_h))
                pre_h=pre_h*pre_gfc2.data.clone().expand_as(pre_h)
                outputs.append(pre_h.view(x.size(0), -1, 1))


        outputs = torch.cat(outputs, dim=-1)
        outputs = self.squash(outputs)
        outputs = outputs.transpose(2,1)
        #****************************************
        #transfer capsule
        #****************************************
        # outputs_list = list(torch.unbind(outputs,dim=-1))
        # aspect_v = outputs_list[0]
        # aspect_v= aspect_v.view(batch_size,transformer_args.max_seq_length,transformer_args.bert_hidden_size)
        #
        #
        # aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3[t]]  # [(N,Co,L), ...]*len(Ks)
        # aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        # aspect_v = torch.cat(aa, 1)
        # # print('aspect_v: ',aspect_v.size()) [3,300]
        #
        #
        # for pre_t in range(1,t+1):
        #     feature = outputs_list[pre_t]
        #     feature= feature.view(batch_size,transformer_args.max_seq_length,transformer_args.bert_hidden_size)
        #     z = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1[t]]  # [(N,Co,L), ...]*len(Ks)
        #     y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect[t](aspect_v).unsqueeze(2)) for conv in self.convs2[t]]
        #     z = [i*j for i, j in zip(z, y)]
        #     z = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in z]  # [(N,Co), ...]*len(Ks)
        #     z = torch.cat(z, 1)
        #     z= z.view(batch_size,1,self.Co*3)
        #     outputs_list[pre_t] = z

        #****************************************
        #representation capsule
        #****************************************

        # aspect_v= aspect_v.contiguous().view(batch_size,1,self.Co*3)
        # outputs_list[0] = aspect_v
        #
        # outputs = torch.cat(outputs_list, dim=1)

        priors = outputs[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
        logits = torch.zeros(*priors.size()).cuda()
        mask=torch.zeros(transformer_args.ntasks).data.cuda()
        for x_id in range(transformer_args.ntasks):
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

        h_output = vote_outputs.view(batch_size,transformer_args.max_seq_length,-1)


        h_output= self.larger(h_output)
        h_output=h_output*glarger.expand_as(h_output)


        return {'outputs':x + h_output,'margin_loss':0}


    def mask(self,t,s=1): #different capsules, different masks
        gfc1=self.gate(s*self.efc1(torch.LongTensor([t]).cuda()))
        gfc2=self.gate(s*self.efc2(torch.LongTensor([t]).cuda()))
        glarger=self.gate(s*self.elarger(torch.LongTensor([t]).cuda()))
        return [gfc1,gfc2,glarger]


class BertAdapterCapsuleGrow(BertAdapter):
    def __init__(self, config):
        super().__init__(config)

        self.num_iterations = 3

        self.gelu = torch.nn.GELU()

        #****************************************
        #Task capsule
        #****************************************

        self.fc1=\
            torch.nn.ModuleList([torch.nn.Linear(config.hidden_size,transformer_args.adapter_size) for _ in range(transformer_args.ntasks)])
        self.fc2=\
            torch.nn.ModuleList([torch.nn.Linear(transformer_args.adapter_size,config.hidden_size) for _ in range(transformer_args.ntasks)])

        #****************************************
        #Transfer capsule
        #****************************************

        D = transformer_args.bert_hidden_size
        self.Co = transformer_args.cnn_kernel_size
        Ks = [3,4,5]

        #TODO: still need to avoid forgetting, next need to consider how to use mask to avoid
        self.convs1 = nn.ModuleList([nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks]) for _ in range(transformer_args.ntasks)])
        self.convs2 = nn.ModuleList([nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks]) for _ in range(transformer_args.ntasks)])
        self.convs3 = nn.ModuleList([nn.ModuleList([nn.Conv1d(D, self.Co, K, padding=K-2) for K in Ks]) for _ in range(transformer_args.ntasks)])
        self.fc_aspect = nn.ModuleList([nn.Linear(self.Co*3, self.Co) for _ in range(transformer_args.ntasks)])

        #****************************************
        #Representation capsule
        #****************************************

        self.route_weights=\
                torch.nn.ParameterList([
                    nn.Parameter(torch.randn(transformer_args.num_share_cap,
                                             transformer_args.ntasks,
                                             self.Co*3,
                                             transformer_args.share_cap_size))
                    for _ in range(transformer_args.ntasks)])

        print('BertAdapterCapsuleGrow')

    def forward(self,x,t):

        #****************************************
        #task capsule
        #****************************************

        batch_size = x.size(0)
        h=self.activation(self.fc1[t](x))
        h=self.activation(self.fc2[t](h))


        outputs = []
        outputs.append(h.view(batch_size, -1, 1))

        with torch.no_grad():
            for pre_t in range(t):
                pre_h=self.activation(self.fc1[pre_t](x)).detach()
                pre_h=self.activation(self.fc2[pre_t](pre_h)).detach()
                outputs.append(pre_h.view(batch_size, -1, 1).detach())

        outputs = torch.cat(outputs, dim=-1)
        outputs = self.squash(outputs)

        #****************************************
        #transfer capsule
        #****************************************
        outputs_list = list(torch.unbind(outputs,dim=-1))
        aspect_v = outputs_list[0]
        aspect_v= aspect_v.view(batch_size,transformer_args.max_seq_length,transformer_args.bert_hidden_size)


        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3[t]]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)
        # print('aspect_v: ',aspect_v.size()) [3,300]

        #TODO: more interation, make sure there will be useful information.

        for pre_t in range(1,t+1):
            feature = outputs_list[pre_t]
            feature= feature.view(batch_size,transformer_args.max_seq_length,transformer_args.bert_hidden_size)
            z = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1[t]]  # [(N,Co,L), ...]*len(Ks)
            y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect[t](aspect_v).unsqueeze(2)) for conv in self.convs2[t]]
            z = [i*j for i, j in zip(z, y)]
            z = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in z]  # [(N,Co), ...]*len(Ks)
            z = torch.cat(z, 1)
            z= z.view(batch_size,1,self.Co*3)
            outputs_list[pre_t] = z

        #****************************************
        #representation capsule
        #****************************************

        aspect_v= aspect_v.contiguous().view(batch_size,1,self.Co*3)
        outputs_list[0] = aspect_v

        outputs = torch.cat(outputs_list, dim=1)

        # task
        # sharing
        priors = outputs[None, :, :, None, :] @ (self.route_weights[t][:,:t+1,:,:])[:, None, :, :, :] # incremental growing
        output,vote_outputs,probs = self.dynamic_routing_instance_based(t,priors)
        vote_outputs = vote_outputs.view(batch_size,1,transformer_args.bert_hidden_size)
        # print('vote_outputs: ',vote_outputs.size())
        # print('x: ',x.size())
        # print('h: ',h.size())

        return x + h + vote_outputs # skip_connect+hat+capsule
        # return x + vote_outputs # worse

    def dynamic_routing_instance_based(self,t,priors):
        priors = priors.squeeze(-2) #[20, 64, 2, 16]

        logits = torch.zeros(*priors.size()[:-1]).cuda()#size: [20, 64, 2]

        for i in range(self.num_iterations):

            probs_tsv = F.softmax(logits, dim=2) #or dim=0

            probs = probs_tsv.unsqueeze(-1).repeat(1, 1, 1,transformer_args.share_cap_size)

            vote_outputs = (probs * priors).sum(dim=2) #voted
            outputs = self.squash(vote_outputs)

            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs[:, :, None, :]).sum(dim=-1)
                logits = logits + delta_logits
        return outputs,vote_outputs,probs_tsv

