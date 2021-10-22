import sys
import torch
import math
import utils
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(torch.nn.Module):

    def __init__(self,taskcla,args):
        super(Net,self).__init__()

        ncha = args.image_channel
        size = args.image_size


        self.taskcla=taskcla
        self.args=args

        pdrop1=0.2
        pdrop2=0.5

        self.drop1=torch.nn.Dropout(pdrop1)
        self.drop2=torch.nn.Dropout(pdrop2)
        self.gate=torch.nn.Sigmoid()

        self.mcl = MainContinualLearning(taskcla,args)
        self.transfer = TransferLayer(taskcla,args)
        self.kt = KnowledgeTransfer(ncha,size,self.taskcla,args)
        self.smax = args.smax
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        print('CNN CAT')
        print('pdrop1: ',pdrop1)
        print('pdrop2: ',pdrop2)

        """ (e.g., used with compression experiments)
        lo,hi=0,2
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        self.efc3.weight.data.uniform_(lo,hi)
        #"""

        return

    def mcl_feature(self,x,gc1,gc2,gc3,gfc1,gfc2):
        h=self.maxpool(self.drop1(self.relu(self.mcl.c1(x))))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop1(self.relu(self.mcl.c2(h))))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop2(self.relu(self.mcl.c3(h))))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.mcl.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.mcl.fc2(h)))
        h=h*gfc2.expand_as(h)
        return h


    # progressive style
    def forward(self,t,x,s=1,phase=None,
                pre_mask=None, pre_task=None,
                similarity=None,history_mask_pre=None,check_federated=None):
        output_dict = {}

        if 'mcl' in phase and 'no_attention' in self.args.loss_type:
            max_masks=self.mask(t,s=s)
            gc1,gc2,gc3,gfc1,gfc2=max_masks
            h = self.mcl_feature(x,gc1,gc2,gc3,gfc1,gfc2)

            if 'dil' in self.args.scenario:
                y = self.mcl.mask_last(h)
            elif 'til' in self.args.scenario:
                y=[]
                for t,i in self.taskcla:
                    y.append(self.mcl.mask_last[t](h))

            output_dict['y'] = y
            output_dict['masks'] = max_masks

            return output_dict


        elif 'mcl' in phase and 'multi-loss-joint-Tsim' in self.args.loss_type:
            max_masks=self.mask(t,s=s)
            gc1,gc2,gc3,gfc1,gfc2=max_masks

            h = self.mcl_feature(x,gc1,gc2,gc3,gfc1,gfc2)

            pre_models = []

            pre_ts= []
            for pre_t in range(t):
                if self.training ==True and similarity[pre_t] :
                    continue
                elif self.training ==False and check_federated.check_t(pre_t)==False:
                    continue

                gc1,gc2,gc3,gfc1,gfc2=self.mask(pre_t,s=self.smax)

                pre_gc1 = gc1.data.clone()
                pre_gc2 = gc2.data.clone()
                pre_gc3 = gc3.data.clone()
                pre_gfc1 = gfc1.data.clone()
                pre_gfc2 = gfc2.data.clone()

                pre_h = self.mcl_feature(x,pre_gc1,pre_gc2,pre_gc3,pre_gfc1,pre_gfc2)

                pre_models.append(pre_h.clone())
                pre_ts.append(pre_t)
            #Tsim: model for each Tsim

            if len(pre_models) > 1:
                task_models = torch.stack(pre_models)
                task_models = task_models.permute(1, 0, 2)

                query = torch.unsqueeze(self.relu(self.kt.q1(torch.LongTensor([t]).cuda())).expand(task_models.size(0),-1),1) #hard to train

                h_attn,_ = self.kt.encoder(task_models,query)

                if 'dil' in self.args.scenario:
                    y = self.mcl.mask_last(h)
                    y_attn = self.mcl.att_last(h)

                elif 'til' in self.args.scenario:
                    y_attn=[]
                    y=[]
                    for t,i in self.taskcla:
                        y.append(self.mcl.mask_last[t](h))
                        y_attn.append(self.mcl.att_last[t](h_attn))

                output_dict['y'] = y
                output_dict['y_attn'] = y_attn
                output_dict['masks'] = max_masks

                return output_dict

            else:

                if 'no-isolate' in self.args.loss_type:
                    if 'dil' in self.args.scenario:
                        y = self.mcl.mask_last(h)
                    elif 'til' in self.args.scenario:
                        y_attn=[]
                        y=[]
                        for t,i in self.taskcla:
                            y.append(self.mcl.mask_last[t](h))

                    output_dict['y'] = y
                    output_dict['y_attn'] = y_attn
                    output_dict['masks'] = max_masks
                    return output_dict

                else:

                    gc1,gc2,gc3,gfc1,gfc2=self.Tsim_mask(t,history_mask_pre=history_mask_pre,similarity=similarity)

                    h_attn = self.mcl_feature(x,gc1,gc2,gc3,gfc1,gfc2)

                    if 'dil' in self.args.scenario:
                        y = self.mcl.mask_last(h)
                        y_attn = self.mcl.att_last(h)

                    elif 'til' in self.args.scenario:
                        y_attn=[]
                        y=[]
                        for t,i in self.taskcla:
                            y.append(self.mcl.mask_last[t](h))
                            y_attn.append(self.mcl.att_last[t](h_attn))

                    output_dict['y'] = y
                    output_dict['y_attn'] = y_attn
                    output_dict['masks'] = max_masks


                    return output_dict


        elif phase == 'transfer':
            gc1,gc2,gc3,gfc1,gfc2=pre_mask
            h = self.mcl_feature(x,gc1,gc2,gc3,gfc1,gfc2)


            if 'dil' in self.args.scenario:
                y = self.transfer.transfer[pre_task][t](self.mcl.mask_last(h))

            elif 'til' in self.args.scenario:
                y=[]
                for t,i in self.taskcla:
                    y.append(self.transfer.transfer[pre_task][t](self.mcl.mask_last[pre_task](h)))

            output_dict['y'] = y

            return output_dict


        elif phase == 'reference':
            gc1,gc2,gc3,gfc1,gfc2=pre_mask
            h = self.mcl_feature(x,gc1,gc2,gc3,gfc1,gfc2)

            if 'dil' in self.args.scenario:
                y = self.transfer.transfer[pre_task][t](self.mcl.mask_last(h))

            elif 'til' in self.args.scenario:
                y=[]
                for t,i in self.taskcla:
                    y.append(self.transfer.transfer[pre_task][t](self.transfer.last[pre_task](h)))

            output_dict['y'] = y

            return output_dict


    def mask(self,t,s=1,phase=None):
        #used by training

        gc1=self.gate(s*self.mcl.ec1(torch.LongTensor([t]).cuda()))
        gc2=self.gate(s*self.mcl.ec2(torch.LongTensor([t]).cuda()))
        gc3=self.gate(s*self.mcl.ec3(torch.LongTensor([t]).cuda()))
        gfc1=self.gate(s*self.mcl.efc1(torch.LongTensor([t]).cuda()))
        gfc2=self.gate(s*self.mcl.efc2(torch.LongTensor([t]).cuda()))
        return [gc1,gc2,gc3,gfc1,gfc2]

    def Tsim_mask(self,t, history_mask_pre=None,similarity=None,phase=None):
        #find the distinct mask, used by block the backward pass

        #want aggregate Tsim
        if phase is None:
           #Tsim mask
            Tsim_gc1=torch.ones_like(self.gate(0*self.mcl.ec1(torch.LongTensor([t]).cuda())))
            Tsim_gc2=torch.ones_like(self.gate(0*self.mcl.ec2(torch.LongTensor([t]).cuda())))
            Tsim_gc3=torch.ones_like(self.gate(0*self.mcl.ec3(torch.LongTensor([t]).cuda())))
            Tsim_gfc1=torch.ones_like(self.gate(0*self.mcl.efc1(torch.LongTensor([t]).cuda())))
            Tsim_gfc2=torch.ones_like(self.gate(0*self.mcl.efc2(torch.LongTensor([t]).cuda())))


        for history_t in range(t):
            if history_t == 0:
                Tsim_gc1_index = history_mask_pre[history_t][0].round().nonzero()
                Tsim_gc2_index = history_mask_pre[history_t][1].round().nonzero()
                Tsim_gc3_index = history_mask_pre[history_t][2].round().nonzero()
                Tsim_gfc1_index = history_mask_pre[history_t][3].round().nonzero()
                Tsim_gfc2_index = history_mask_pre[history_t][4].round().nonzero()
            else:
                Tsim_gc1_index = (history_mask_pre[history_t][0] - history_mask_pre[history_t-1][0]).round().nonzero()
                Tsim_gc2_index = (history_mask_pre[history_t][1] - history_mask_pre[history_t-1][1]).round().nonzero()
                Tsim_gc3_index = (history_mask_pre[history_t][2] - history_mask_pre[history_t-1][2]).round().nonzero()
                Tsim_gfc1_index = (history_mask_pre[history_t][3] - history_mask_pre[history_t-1][3]).round().nonzero()
                Tsim_gfc2_index = (history_mask_pre[history_t][4] - history_mask_pre[history_t-1][4]).round().nonzero()
            if similarity[history_t]==0:
                Tsim_gc1[Tsim_gc1_index[:,0],Tsim_gc1_index[:,1]] = 0
                Tsim_gc2[Tsim_gc2_index[:,0],Tsim_gc2_index[:,1]] = 0
                Tsim_gc3[Tsim_gc3_index[:,0],Tsim_gc3_index[:,1]] = 0
                Tsim_gfc1[Tsim_gfc1_index[:,0],Tsim_gfc1_index[:,1]] = 0
                Tsim_gfc2[Tsim_gfc2_index[:,0],Tsim_gfc2_index[:,1]] = 0

        return [Tsim_gc1,Tsim_gc2,Tsim_gc3,Tsim_gfc1,Tsim_gfc2]



    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks

        if n=='mcl.fc1.weight':
            return gfc1.data.view(-1,1).expand_as(self.mcl.fc1.weight)
        elif n=='mcl.fc1.bias':
            return gfc1.data.view(-1)
        elif n=='mcl.fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.mcl.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.mcl.fc2.weight)
            return torch.min(post,pre)
        elif n=='mcl.fc2.bias':
            return gfc2.data.view(-1)
        elif n=='mcl.c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.mcl.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.mcl.c2.weight)
            return torch.min(post,pre)
        elif n=='mcl.c2.bias':
            return gc2.data.view(-1)
        elif n=='mcl.c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.mcl.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.mcl.c3.weight)
            return torch.min(post,pre)
        elif n=='mcl.c3.bias':
            return gc3.data.view(-1)
        return None



class MainContinualLearning(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(MainContinualLearning, self).__init__()

        ncha = args.image_channel
        size = args.image_size
        self.taskcla = taskcla
        self.args = args

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
            self.mask_last=torch.nn.Linear(2048,args.nclasses)
            self.att_last=torch.nn.Linear(2048,args.nclasses)

        elif 'til' in args.scenario:
            self.mask_last=torch.nn.ModuleList()
            self.att_last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.mask_last.append(torch.nn.Linear(2048,n))
                self.att_last.append(torch.nn.Linear(2048,n))


class TransferLayer(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(TransferLayer, self).__init__()
        self.taskcla = taskcla
        ncha = args.image_channel
        size = args.image_size
        self.args = args


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

        
        self.fusion = torch.nn.Linear(2048*2,2048)

        if 'dil' in self.args.scenario:
            self.last=torch.nn.Linear(2048,self.args.nclasses)
            self.last_fusion=torch.nn.Linear(2048*2,self.args.nclasses)

            # this one will not change according to different scenario
            self.transfer=torch.nn.ModuleList()
            for from_t,from_n in taskcla:
                self.transfer_to_n=torch.nn.ModuleList()
                for to_t,to_n in taskcla:
                    self.transfer_to_n.append(torch.nn.Linear(self.args.nclasses,self.args.nclasses))
                self.transfer.append(self.transfer_to_n)


        elif 'til' in self.args.scenario:
            self.last=torch.nn.ModuleList()
            self.last_fusion=torch.nn.ModuleList()

            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(2048,n))
                self.last_fusion.append(torch.nn.Linear(2048*2,n))

            self.transfer=torch.nn.ModuleList()
            for from_t,from_n in taskcla:
                self.transfer_to_n=torch.nn.ModuleList()
                for to_t,to_n in taskcla:
                    self.transfer_to_n.append(torch.nn.Linear(from_n,to_n))
                self.transfer.append(self.transfer_to_n)

class KnowledgeTransfer(torch.nn.Module):

    def __init__(self,ncha,size,taskcla,args):

        super(KnowledgeTransfer, self).__init__()
        nhid = 2048
        #self-attention ==============
        self.q1=torch.nn.Embedding(len(taskcla),nhid)
        self.encoder = EncoderLayer(args.n_head, nhid, nhid, int(nhid/args.n_head), int(nhid/args.n_head), args=args)
        # n_head, d_model, d_k, d_v


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_head, d_model, d_inner, d_k, d_v, dropout=0.1,args=None):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.position_enc = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.args=args
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, enc_q=None,ranking=None):
        #TODO: Positional/ranking embedding

        if enc_q is None:
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
            enc_output = self.pos_ffn(enc_output)

        else:
            enc_output, enc_slf_attn = self.slf_attn(enc_q, enc_input, enc_input)
            enc_output = self.pos_ffn(enc_output)

        enc_output = self.layer_norm(enc_output)

        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5) #sqrt d_k

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v):


        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = torch.squeeze(q,1)
        q = self.layer_norm(q)


        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)

        if len_q == 1:
            q = q.transpose(1, 2).contiguous().view(sz_b,-1)
        else:
            q = q.transpose(1, 2).contiguous().view(sz_b, len_q,-1)

        q = self.dropout(self.fc(q))
        q += residual

        return q, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=40):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, enc_input,ranking):
        return enc_input + self.pos_table[:, ranking].clone().detach()

