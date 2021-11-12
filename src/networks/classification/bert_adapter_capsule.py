#coding: utf-8
import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn

sys.path.append("./networks/base/")
from my_transformers import MyBertModel

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()
        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        args.build_adapter_capsule = True
        self.bert = MyBertModel.from_pretrained(args.bert_model,config=config,args=args)

        for param in self.bert.parameters():
            # param.requires_grad = True
            param.requires_grad = False

        adapter_masks = \
            [self.bert.encoder.layer[layer_id].attention.output.adapter_capsule for layer_id in range(config.num_hidden_layers)] + \
            [self.bert.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
            [self.bert.encoder.layer[layer_id].output.adapter_capsule for layer_id in range(config.num_hidden_layers)] + \
            [self.bert.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        for adapter_mask in adapter_masks:
            for param in adapter_mask.parameters():
                param.requires_grad = True
                # param.requires_grad = False

        self.taskcla=taskcla
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(args.bert_hidden_size,n))
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args
        self.config = config
        self.num_task = len(taskcla)

        self.apply_bert_attention_output = args.apply_bert_attention_output
        self.apply_bert_output = args.apply_bert_output

        print('BERT ADAPTER CAPSULE')

        return

    def forward(self,t,input_ids, segment_ids, input_mask, targets, s=1):

        output_dict = {}


        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,t=t,s=s)

        pooled_output = self.dropout(pooled_output)
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](pooled_output))

        masks = self.mask(t,s)

        output_dict['y'] = y
        output_dict['masks'] = masks
        output_dict['margin_loss'] = 0

        return output_dict

    def mask(self,t,s):
        masks = {}
        for layer_id in range(self.config.num_hidden_layers):
            if self.apply_bert_output:
                if self.args.transfer_route:
                    key = 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.transfer_capsules.larger' #gfc1
                    masks[key] = self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.transfer_capsules.mask(t,s)
                else:
                    key = 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.tsv_capsules.larger' #gfc1
                    masks[key] = self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.mask(t,s)

            if self.apply_bert_attention_output:

                if self.args.transfer_route:
                    key = 'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.transfer_capsules.larger' #gfc1
                    masks[key] = self.bert.encoder.layer[layer_id].attention.output.adapter_capsule.capsule_net.transfer_capsules.mask(t,s)
                else:
                    key = 'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.tsv_capsules.larger' #gfc1
                    masks[key] = self.bert.encoder.layer[layer_id].attention.output.adapter_capsule.capsule_net.tsv_capsules.mask(t,s)

        return masks



    def get_view_for(self,n,p,masks):

        for layer_id in range(self.config.num_hidden_layers):

            if self.args.transfer_route:
                if n == \
                'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.transfer_capsules.larger.weight': #gfc1
                    # print('transfer_capsules not none')
                    return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                elif n == \
                'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.transfer_capsules.larger.bias': #gfc1
                    return masks[n.replace('.bias','')].data.view(-1)

                if n == \
                'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.transfer_capsules.larger.weight': #gfc1
                    # print('transfer_capsules not none')
                    return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                elif n == \
                'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.transfer_capsules.larger.bias': #gfc1
                    return masks[n.replace('.bias','')].data.view(-1)
            else:
                if n == \
                'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.tsv_capsules.larger.weight': #gfc1
                    # print('tsv_capsules not none')
                    return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                elif n == \
                'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.tsv_capsules.larger.bias': #gfc1
                    return masks[n.replace('.bias','')].data.view(-1)

                if n == \
                'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.tsv_capsules.larger.weight': #gfc1
                    # print('tsv_capsules not none')
                    return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                elif n == \
                'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.tsv_capsules.larger.bias': #gfc1
                    return masks[n.replace('.bias','')].data.view(-1)


        return None


        return None

    def get_view_for_tsv(self,n,t):
        #TODO: Cautions! Don't preint, this is used in forward transfer DONT USE TSV_capsules for route version
        for layer_id in range(self.config.num_hidden_layers):
            if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.tsv_capsules.route_weights':
                # print('not none')
                return self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.tsv[t].data.view(1,-1,1,1)
            # if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.transfer_capsules.route_weights':
            #     # print('not none')
            #     return self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.tsv[t].data.view(1,-1,1,1)

            for c_t in range(self.num_task):
                if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight':
                    # print('attention semantic_capsules fc1')
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias':
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data

                elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight':
                    # print('not none')
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias':
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data

                if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.weight':
                    # print('attention semantic_capsules fc1')
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.bias':
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data

            if not self.args.unblock_attention:
                if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.tsv_capsules.route_weights':
                    # print('not none')
                    return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule.capsule_net.tsv_capsules.tsv[t].data.view(1,-1,1,1)
                # if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.transfer_capsules.route_weights':
                #     # print('not none')
                #     return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule.capsule_net.tsv_capsules.tsv[t].data.view(1,-1,1,1)

                for c_t in range(self.num_task):
                    if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight':
                        # print('attention semantic_capsules fc1')
                        return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data
                    elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias':
                        return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data

                    elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight':
                        # print('not none')
                        return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data
                    elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias':
                        return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data

                    if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.weight':
                        # print('attention semantic_capsules fc1')
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data
                    elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.bias':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule.capsule_net.tsv_capsules.tsv[t][c_t].data


        return 1 #if no condition is satified