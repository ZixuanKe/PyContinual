#coding: utf-8
import sys
import torch
from transformers import  BertConfig
import utils
from torch import nn
import torch.nn.functional as F

#TODO: I want capsule on top of BERT

from transformers import BertModel

sys.path.append("./networks/base/")
from my_transformers import MyBertModel

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()
        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False

        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model,config=config) #nomral bert here, and fixed weight
        if args.use_imp:
            from adapters import BertAdapterCapsuleMaskImp as BertAdapterCapsuleMask
            from adapters import BertAdapterCapsuleImp as BertAdapterCapsule
        else:
            from adapters import BertAdapterCapsuleMask
            from adapters import BertAdapterCapsule

        self.taskcla=taskcla
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.adapter_capsule_mask = BertAdapterCapsuleMask(args)

        # BERT fixed, i.e. BERT as feature extractor
        # not adapter_capsule is still trainable
        for param in self.bert.parameters():
            param.requires_grad = False

        self.num_task = len(taskcla)
        self.num_kernel = 3
        self.config = config

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(config.hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(config.hidden_size,n))


        print('DIL BERT')

        return

    def forward(self,t,input_ids, segment_ids, input_mask,targets,s):

        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        output_dict = self.adapter_capsule_mask(sequence_output, t, s, targets)
        hidden_states = output_dict['outputs']


        pooled_output = self.dropout(torch.mean(hidden_states,1))


        if 'dil' in self.args.scenario:
            y = self.last(pooled_output)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](pooled_output))

        output_dict['y'] = y
        masks = self.mask(t,s)
        output_dict['masks'] = masks

        return output_dict



    def mask(self,t,s):
        # TSM and also the larger, all larger needs the mask
        # we use dict to remember the layer that needs masked
        masks = {}

        fc1_key = 'adapter_capsule_mask.fc1'  # gfc1
        fc2_key = 'adapter_capsule_mask.fc2'  # gfc2

        masks[fc1_key], masks[fc2_key] = self.adapter_capsule_mask.mask(t, s)

        key = 'adapter_capsule_mask.capsule_net.tsv_capsules.larger'  # gfc1
        masks[key] = self.adapter_capsule_mask.capsule_net.tsv_capsules.mask(t, s)

        return masks



    def get_view_for(self,n,p,masks):

        if n == 'adapter_capsule_mask.fc1.weight':
            return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
        elif n == 'adapter_capsule_mask.fc1.bias':
            return masks[n.replace('.bias', '')].data.view(-1)
        elif n == 'adapter_capsule_mask.fc2.weight':
            post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
            return torch.min(post, pre)
        elif n == 'adapter_capsule_mask.fc2.bias':
            return masks[n.replace('.bias', '')].data.view(-1)


        if n == 'adapter_capsule_mask.capsule_net.tsv_capsules.larger.weight': #gfc1
            # print('tsv_capsules not none')
            return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
        elif n == 'adapter_capsule_mask.capsule_net.tsv_capsules.larger.bias': #gfc1
            return masks[n.replace('.bias','')].data.view(-1)


        return None




    def get_view_for_tsv(self,n,t):
        #TODO: Cautions! Don't preint, this is used in forward transfer
        if n=='adapter_capsule_mask.capsule_net.tsv_capsules.route_weights':
            # print('not none')
            return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t].data.view(1,-1,1,1)
        for c_t in range(self.num_task):
            if n=='adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight':
                # print('attention semantic_capsules fc1')
                return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
            elif n=='adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias':
                return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

            elif n=='adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight':
                # print('not none')
                return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
            elif n=='adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias':
                return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

            if n=='adapter_capsule_mask.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.weight':
                # print('attention semantic_capsules fc1')
                return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
            elif n=='adapter_capsule_mask.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.bias':
                return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

            for m_t in range(self.num_kernel):
                if n=='adapter_capsule_mask.capsule_net.transfer_capsules.convs3.'+str(c_t)+'.'+str(m_t)+'.weight':
                    return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n=='adapter_capsule_mask.capsule_net.transfer_capsules.convs3.'+str(c_t)+'.'+str(m_t)+'.bias':
                    return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n=='adapter_capsule_mask.capsule_net.transfer_capsules.convs2.'+str(c_t)+'.'+str(m_t)+'.weight':
                    return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n=='adapter_capsule_mask.capsule_net.transfer_capsules.convs2.'+str(c_t)+'.'+str(m_t)+'.bias':
                    return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n=='adapter_capsule_mask.capsule_net.transfer_capsules.convs1.'+str(c_t)+'.'+str(m_t)+'.weight':
                    return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n=='adapter_capsule_mask.capsule_net.transfer_capsules.convs1.'+str(c_t)+'.'+str(m_t)+'.bias':
                    # print('not none')
                    return self.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

        return 1 #if no condition is satified