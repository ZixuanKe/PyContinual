
import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsNetBCL(nn.Module):
    def __init__(self,input_size,down_sample,args):
        super().__init__()
        self.semantic_capsules = CapsuleLayerBCL(input_size,down_sample,args,'semantic')
        self.tsv_capsules = CapsuleLayerBCL(input_size,down_sample,args,'tsv')
        self.args = args
        self.input_size = input_size
        self.down_sample = down_sample


    def forward(self, x):
        semantic_output = self.semantic_capsules(x,'semantic')
        tsv_output = self.tsv_capsules(semantic_output,'tsv')
        return tsv_output



class CapsuleLayerBCL(nn.Module): #it has its own number of capsule for output
    def __init__(self, input_size,down_sample,args,layer_type):
        super().__init__()

        self.input_size = input_size
        self.down_sample = down_sample

        if layer_type=='tsv':
            self.num_routes = args.ntasks
            self.num_capsules = args.semantic_cap_size
            self.class_dim = args.max_source_length
            self.in_channel = args.max_source_length*args.semantic_cap_size
            # self.in_channel = 100

            # self.larger=torch.nn.Linear(args.semantic_cap_size,1024) #each task has its own larger way
            self.larger = nn.ModuleList([torch.nn.Linear(args.semantic_cap_size,self.down_sample) for _ in range(args.ntasks)])  # each task has its own larger way
            self.gate=torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax()
            self.num_iterations = 3
            self.route_weights = \
                nn.Parameter(torch.randn(self.num_capsules, self.num_routes, self.in_channel, self.class_dim))

            self.tsv = torch.tril(torch.ones(args.ntasks,args.ntasks)).data.cuda()# for backward

        elif layer_type=='semantic':
            self.fc1 = nn.ModuleList([torch.nn.Linear(self.input_size, self.down_sample) for _ in range(args.ntasks)])
            self.fc2 = nn.ModuleList([torch.nn.Linear(self.down_sample, args.semantic_cap_size) for _ in range(args.ntasks)])

        self.args= args

    def forward(self, x,layer_type=None):
        if layer_type=='tsv':
            batch_size = x.size(0)
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = torch.zeros(*priors.size()).cuda()
            mask=torch.zeros(self.args.ntasks).data.cuda()
            for x_id in range(self.args.ntasks):
                if self.tsv[self.args.eval_t][x_id] == 0: mask[x_id].fill_(-10000) # block future, all previous are the same

            for i in range(self.num_iterations):
                logits = logits*self.tsv[self.args.eval_t].data.view(1,1,-1,1,1) #multiply 0 to future task
                logits = logits + mask.data.view(1,1,-1,1,1) #add a very small negative number
                probs = self.my_softmax(logits, dim=2)
                vote_outputs = (probs * priors).sum(dim=2, keepdim=True) #voted
                outputs = self.squash(vote_outputs)

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits

            h_output = vote_outputs.view(batch_size,self.args.max_source_length,-1)

            # h_output= self.larger(h_output)
            h_output = self.larger[self.args.eval_t](h_output)

            return h_output

        elif layer_type=='semantic':

            outputs = [fc2(fc1(x)).view(x.size(0), -1, 1) for fc1, fc2 in zip(self.fc1, self.fc2)]
            outputs = torch.cat(outputs, dim=-1)

            outputs = self.squash(outputs)
            return outputs.transpose(2,1)

    def my_softmax(self,input, dim=1):
        transposed_input = input.transpose(dim, len(input.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


    def squash(self, input_tensor, dim=-1,epsilon=1e-16): # 0 will happen in our case, has to add an epsilon
        squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
        squared_norm = squared_norm + epsilon
        scale = squared_norm / (1 + squared_norm)
        return scale * input_tensor / torch.sqrt(squared_norm)


class CapsNetCTR(nn.Module):
    def __init__(self,input_size,down_sample,args):
        super().__init__()
        self.input_size = input_size
        self.down_sample = down_sample
        self.semantic_capsules = CapsuleLayerCTR(input_size,down_sample,args,'semantic')
        self.transfer_capsules = CapsuleLayerCTR(input_size,down_sample,args,'transfer_route')
        self.tsv_capsules = CapsuleLayerCTR(input_size,down_sample,args,'tsv')
        self.args = args

    def forward(self, x):
        max_length = x.size(1) # decoder and encoder will have different size
        semantic_output = self.semantic_capsules(x,max_length,'semantic')
        transfer_output = self.transfer_capsules(semantic_output,max_length,'transfer_route')
        tsv_output = self.tsv_capsules(transfer_output,max_length,'tsv')
        return tsv_output

class CapsuleLayerCTR(nn.Module): #it has its own number of capsule for output
    def __init__(self, input_size,down_sample,args,layer_type):
        super().__init__()
        self.input_size = input_size
        self.down_sample = down_sample
        if layer_type=='semantic':

            self.fc1 = nn.ModuleList([torch.nn.Linear(self.input_size, self.down_sample) for _ in range(args.ntasks)])
            self.fc2 = nn.ModuleList( [torch.nn.Linear(self.down_sample, args.semantic_cap_size) for _ in range(args.ntasks)])

            self.gelu=torch.nn.GELU()

        elif layer_type=='tsv':
            self.num_routes = args.ntasks
            self.num_capsules = args.semantic_cap_size
            self.class_dim = args.max_source_length
            self.in_channel = args.max_source_length*args.semantic_cap_size
            # self.in_channel = 100
            self.gate=torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax()
            self.num_iterations = 3

            #no routing for max_source_length
            self.route_weights = \
                nn.Parameter(torch.randn(args.num_semantic_cap, self.num_routes, args.semantic_cap_size, args.semantic_cap_size))

            self.tsv = torch.tril(torch.ones(args.ntasks,args.ntasks)).data.cuda()# for backward



        elif layer_type=='transfer_route':

            D = args.semantic_cap_size*args.num_semantic_cap
            self.Co = 100

            Ks = [3,4,5]

            self.len_ks = len(Ks)

            # self.convs1 = nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks])
            self.convs2 = nn.ModuleList([nn.Conv1d(D, self.Co, K) for K in Ks])
            self.convs3 = nn.ModuleList([nn.Conv1d(D, self.Co, K, padding=K-2) for K in Ks])
            self.fc_cur = nn.Linear(self.Co*self.len_ks, self.Co)
            self.fc_sim = nn.Linear(300, args.num_semantic_cap*args.semantic_cap_size)
            self.convs4 =  nn.Conv1d(300, 2, 1)

            self.num_routes = args.ntasks
            self.num_capsules = args.semantic_cap_size
            self.class_dim = args.max_source_length
            self.in_channel = args.max_source_length*args.semantic_cap_size
            # self.in_channel = 100
            self.gate=torch.nn.Sigmoid()

            #no routing for max_source_length
            self.route_weights = \
                nn.Parameter(torch.randn(args.num_semantic_cap, self.num_routes, args.semantic_cap_size, args.semantic_cap_size))

            # self.larger=torch.nn.Linear(args.semantic_cap_size*args.num_semantic_cap,1024) #each task has its own larger way
            self.larger = nn.ModuleList([torch.nn.Linear(args.semantic_cap_size * args.num_semantic_cap, self.down_sample) for _ in range(args.ntasks)])  # each task has its own larger way

            self.tsv = torch.tril(torch.ones(args.ntasks,args.ntasks)).data.cuda()# for backward

        self.args = args

    def forward(self,x,max_length,layer_type=None):

        if layer_type=='semantic':
            outputs = [fc2(fc1(x)).view(x.size(0), -1, 1) for fc1, fc2 in zip(self.fc1, self.fc2)]
            outputs = torch.cat(outputs, dim=-1) #(B,cap_size,19)

            outputs = self.squash(outputs)

            return outputs

        elif layer_type=='transfer_route':
                batch_size = x.size(0)
                x = x.contiguous().view(batch_size*max_length,-1,self.args.semantic_cap_size)

                priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

                outputs_list = list(torch.unbind(priors,dim=2))

                decision_maker = []
                sim_attn = []
                for pre_t in range(self.args.ntasks):

                    #Regarding ASC, for numberical stable, I change relu to gelu
                        cur_v = outputs_list[self.args.eval_t] #current_task
                        cur_v= cur_v.contiguous().view(batch_size,max_length,self.args.num_semantic_cap*self.args.semantic_cap_size)

                        aa = [F.relu(conv(cur_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
                        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
                        cur_v = torch.cat(aa, 1)

                        feature = outputs_list[pre_t]
                        feature= feature.contiguous().view(batch_size,max_length,self.args.num_semantic_cap*self.args.semantic_cap_size)
                        # z = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)

                        # decoder will have max_length=1 when testing, the convolution is not possible in the decoder any more
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

                vote_outputs = (self.tsv[self.args.eval_t].data.view(1,1,-1,1,1) *
                    sim_attn.repeat(max_length,1,1)
                                .view(self.args.num_semantic_cap,-1,self.args.ntasks,1,self.args.semantic_cap_size) *
                    decision_maker.repeat(max_length,1)
                                .view(1,-1,self.args.ntasks,1,1) *
                    priors).sum(dim=2, keepdim=True) #route


                h_output = vote_outputs.view(batch_size,max_length,-1)
                # h_output = self.larger(h_output)
                h_output = self.larger[self.args.eval_t](h_output)

                return h_output

        elif layer_type=='tsv':
            return x



    def my_softmax(self,input, dim=1):
        transposed_input = input.transpose(dim, len(input.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

    def squash(self, input_tensor, dim=-1,epsilon=1e-16): # 0 will happen in our case, has to add an epsilon
        squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
        squared_norm = squared_norm + epsilon
        scale = squared_norm / (1 + squared_norm)
        return scale * input_tensor / torch.sqrt(squared_norm)


class MyAdapter(nn.Module):
    def __init__(self,input_size,down_sample,args):
        super().__init__()

        self.input_size = input_size
        self.down_sample = down_sample

        self.fc1 = torch.nn.Linear(self.input_size, self.down_sample) #bottle net size
        self.fc2 = torch.nn.Linear(self.down_sample, self.down_sample)
        self.activation = torch.nn.GELU()
        self.args = args

        if 'adapter_hat' in self.args.baseline or 'adapter_cat' in self.args.baseline or 'adapter_classic' in self.args.baseline:
            self.efc1=torch.nn.Embedding(args.ntasks,self.down_sample)
            self.efc2=torch.nn.Embedding(args.ntasks,self.down_sample)
            self.gate=torch.nn.Sigmoid()

        elif 'adapter_bcl' in self.args.baseline:
            self.capsule_net = CapsNetBCL(input_size,input_size,args)  # no change at first
            self.efc1=torch.nn.Embedding(args.ntasks,self.down_sample)
            self.efc2=torch.nn.Embedding(args.ntasks,self.down_sample)
            self.gate=torch.nn.Sigmoid()

        elif 'adapter_ctr' in self.args.baseline:
            self.capsule_net = CapsNetCTR(input_size,input_size,args)
            self.efc1=torch.nn.Embedding(args.ntasks,self.down_sample)
            self.efc2=torch.nn.Embedding(args.ntasks,self.down_sample)
            self.gate=torch.nn.Sigmoid()


    def forward(self, x,layer_type):

        if 'adapter_hat' in self.args.baseline or 'adapter_cat' in self.args.baseline or 'adapter_classic' in self.args.baseline:
            gfc1, gfc2 = self.mask()
            h=self.activation(self.fc1(x))
            h=h*gfc1.expand_as(h)

            h=self.activation(self.fc2(h))
            h=h*gfc2.expand_as(h)

        elif 'adapter_bcl' in self.args.baseline:

            if layer_type == 'encoder':
                capsule_output = self.capsule_net(x)
                h = x + capsule_output  # skip-connection
            else:
                h=x # do nothing for decoder

            # task specifc
            gfc1, gfc2 = self.mask()

            h = self.activation(self.fc1(h))
            h = h * gfc1.expand_as(h)

            h = self.activation(self.fc2(h))
            h = h * gfc2.expand_as(h)

        elif 'adapter_ctr' in self.args.baseline:

            if layer_type == 'encoder':
                capsule_output = self.capsule_net(x)
                h = x + capsule_output  # skip-connection
            else:
                h=x # do nothing for decoder

            # task specifc
            gfc1, gfc2 = self.mask()

            h = self.activation(self.fc1(h))
            h = h * gfc1.expand_as(h)

            h = self.activation(self.fc2(h))
            h = h * gfc2.expand_as(h)

        else:
            h = self.activation(self.fc1(x))
            h = self.activation(self.fc2(h))

        # return x + h
        return h # residual takes care by the adapter_transformer package



    def mask(self):

       efc1 = self.efc1(torch.LongTensor([self.args.eval_t]).cuda())
       efc2 = self.efc2(torch.LongTensor([self.args.eval_t]).cuda())

       gfc1=self.gate(self.args.s*efc1)
       gfc2=self.gate(self.args.s*efc2)

       return [gfc1,gfc2]

    def cat_mask(self):
        # print('self.args.similarity in cat mask: ',self.args.similarity.similarities)
        gfc1, gfc2 = self.mask()

        cat_gfc1 = torch.zeros_like(gfc1)
        cat_gfc2 = torch.zeros_like(gfc2)

        cur_t = self.args.eval_t
        for prev_t in range(len(self.args.similarity.similarities)):
            self.args.eval_t = prev_t
            gfc1, gfc2 = self.mask()

            if self.args.similarity.similarities[prev_t] == 1: # similar

                cat_gfc1 = torch.max(cat_gfc1,gfc1)
                cat_gfc2 = torch.max(cat_gfc2,gfc2)

        # we need this becuase later we are going to open it instead of close it
        cat_gfc1 = 1 - cat_gfc1
        cat_gfc2 = 1 - cat_gfc2

        self.args.eval_t = cur_t
        return [cat_gfc1,cat_gfc2]



class Myplugin(nn.Module):
    def __init__(self, input_size, down_sample, args):
        """
        Args are seperated into different sub-args here
        """
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.down_sample = down_sample

        if 'adapter_demix' in args.baseline:
            self.adapters = torch.nn.ModuleList()
            for i in range(args.ntasks): # quite unstable
                self.adapters.append(MyAdapter(input_size,down_sample,args))

        elif  'adapter_ctr' in args.baseline \
                or 'adapter_hat' in args.baseline or 'adapter_cat' in args.baseline \
                or 'adapter_bcl' in args.baseline \
                or 'adapter_classic' in args.baseline:
            self.adapters = MyAdapter(input_size,down_sample,args)


    def forward(self, x,layer_type):

        if 'adapter_demix' in self.args.baseline:
            return self.adapters[self.args.eval_t](x)

        elif  'adapter_ctr' in self.args.baseline \
                or 'adapter_hat' in self.args.baseline \
                or 'adapter_cat' in self.args.baseline\
                or 'adapter_bcl' in self.args.baseline\
                or 'adapter_classic' in self.args.baseline:
            return self.adapters(x,layer_type)

