import sys,time
import numpy as np
import torch
import os
import logging
import glob
import math
import json
import argparse
import random
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
import utils
# from apex import amp
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
import torch.nn.functional as F
import functools
import torch.nn as nn
from copy import deepcopy
sys.path.append("./approaches/")
from loss_metric_zoo import SupConLoss,DistillKL


#TODO: merge with bert_adapter_base.py
class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, logits, target):
        probs = F.softmax(logits, 1)
        loss = (- target * torch.log(probs)).sum(1).mean()
        return loss

class Appr(object):

    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x


    def __init__(self,model,aux_model=None,logger=None,taskcla=None, args=None):
        # can deal with aux and unaux
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        args.output_dir = args.output_dir.replace(
            '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(args.__dict__, open(os.path.join(
            args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        logger.info("device: {} n_gpu: {}".format(
            self.device, self.n_gpu))

        self.clipgrad=10000

        self.aux_model=aux_model
        self.model=model
        self.logger = logger

        self.train_batch_size=args.train_batch_size
        self.eval_batch_size=args.eval_batch_size
        self.args=args
        self.ce=torch.nn.CrossEntropyLoss()
        self.soft_ce=SoftCrossEntropy()
        self.sup_con = SupConLoss(temperature=args.temp,base_temperature=args.base_temp)
        self.kd = DistillKL(4)

        self.smax = 400
        self.thres_cosh=50
        self.thres_emb=6
        self.lamb=0.75

        self.mask_pre=None
        self.mask_back=None
        self.aux_mask_pre=None
        self.aux_mask_back=None


        self.tsv_para = \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.tsv_capsules.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.tsv_capsules.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.tsv_capsules.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)]




        print('DIL BERT ADAPTER MASK BASE')

    def sup_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s):
        if self.args.sup_head:
            outputs = torch.cat([output.clone().unsqueeze(1), output.clone().unsqueeze(1)], dim=1)
        else:
            outputs = torch.cat([pooled_rep.clone().unsqueeze(1), pooled_rep.clone().unsqueeze(1)], dim=1)

        sup_loss = self.sup_con(outputs, targets,args=self.args)
        return sup_loss

    def augment_current_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s):
        bsz = input_ids.size(0)

        idxs,ls=self.idx_generator(bsz)

        ref_pooled_reps = []
        ref_outputs = []
        for idx in idxs: #no need to re-run...
            ref_output = output[idx].clone()
            ref_pooled_rep = pooled_rep[idx].clone()

            ref_pooled_reps.append(ref_pooled_rep)
            ref_outputs.append(ref_output)

        for idx_,idx in enumerate(idxs):
            l = ls[idx_]
            if self.args.current_head:
                outputs = [output.clone().unsqueeze(1)]
                ref_outputs = [ref_output.clone().unsqueeze(1)]

            else:
                outputs = [pooled_rep.clone().unsqueeze(1)]
                ref_outputs = [ref_pooled_rep.clone().unsqueeze(1)]

            pre_output_dict = self.model(t,input_ids, segment_ids, input_mask,s=s,l=l,idx=idx,start_mixup=True,mix_type='tmix') #pre_t as indicator
            pre_pooled_rep = pre_output_dict['normalized_pooled_rep']
            pre_output = pre_output_dict['y']

            if self.args.current_head:
                outputs.append(pre_output.unsqueeze(1).clone())
                ref_outputs.append(pre_output.unsqueeze(1).clone())
            else:
                outputs.append(pre_pooled_rep.unsqueeze(1).clone())
                ref_outputs.append(pre_pooled_rep.unsqueeze(1).clone())

            outputs = torch.cat(outputs, dim=1)
            ref_outputs = torch.cat(ref_outputs, dim=1)

            current_loss = self.sup_con(outputs,targets,args=self.args) #same sampel, different domain, as close as possible
            ref_current_loss = self.sup_con(ref_outputs,targets[idx],args=self.args) #same sampel, different domain, as close as possible
            augment_current_loss = l*current_loss + (1-l)* ref_current_loss

        return augment_current_loss



    def augment_distill_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s):
        bsz = input_ids.size(0)

        if self.args.tmix: #tmix can vbe multiple
            idxs,ls=self.idx_generator(bsz)

            ref_pooled_reps = []
            ref_outputs = []
            for idx in idxs: #no need to re-run...
                ref_output = output[idx].clone()
                ref_pooled_rep = pooled_rep[idx].clone()

                ref_pooled_reps.append(ref_pooled_rep)
                ref_outputs.append(ref_output)

            for idx_,idx in enumerate(idxs):
                l = ls[idx_]
                if self.args.distill_head:
                    outputs = [output.clone().unsqueeze(1)]
                    ref_outputs = [ref_output.clone().unsqueeze(1)]

                else:
                    outputs = [pooled_rep.clone().unsqueeze(1)]
                    ref_outputs = [ref_pooled_rep.clone().unsqueeze(1)]

                with torch.no_grad():
                    for pre_t in range(t):
                        pre_output_dict = self.model(pre_t,input_ids, segment_ids, input_mask,s=self.smax,l=l,idx=idx,start_mixup=True,mix_type='tmix') #pre_t as indicator
                        pre_pooled_rep = pre_output_dict['normalized_pooled_rep']
                        pre_output = pre_output_dict['y']

                if self.args.distill_head:
                    outputs.append(pre_output.unsqueeze(1).clone())
                    ref_outputs.append(pre_output.unsqueeze(1).clone())
                else:
                    outputs.append(pre_pooled_rep.unsqueeze(1).clone())
                    ref_outputs.append(pre_pooled_rep.unsqueeze(1).clone())

                outputs = torch.cat(outputs, dim=1)
                ref_outputs = torch.cat(ref_outputs, dim=1)

                distill_loss = self.sup_con(outputs,args=self.args) #same sampel, different domain, as close as possible
                ref_distill_loss = self.sup_con(ref_outputs,args=self.args) #same sampel, different domain, as close as possible
                augment_distill_loss = l*distill_loss + (1-l)* ref_distill_loss

        else: #no tmix, no need to perprotion
            if self.args.distill_head:
                outputs = [output.clone().unsqueeze(1)]
            else:
                outputs = [pooled_rep.clone().unsqueeze(1)]

            with torch.no_grad():
                for pre_t in range(t):
                    pre_output_dict = self.model(pre_t,input_ids, segment_ids, input_mask,s=self.smax)
                    pre_pooled_rep = pre_output_dict['normalized_pooled_rep']
                    pre_output = pre_output_dict['y']
            if self.args.distill_head:
                outputs.append(pre_output.unsqueeze(1).clone())
            else:
                outputs.append(pre_pooled_rep.unsqueeze(1).clone())
            outputs = torch.cat(outputs, dim=1)
            augment_distill_loss= self.sup_con(outputs,args=self.args)

        return augment_distill_loss


    def amix_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s):
        amix_loss = 0
        if self.args.amix_head:
            mix_pooled_reps = [output.clone().unsqueeze(1)]
        else:
            mix_pooled_reps = [pooled_rep.clone().unsqueeze(1)]

        if self.args.attn_type == 'cos':
            for n in range(self.args.naug+1):
                mix_output_dict = self.model(t,input_ids, segment_ids, input_mask,s=s,start_mixup=True,l=n,mix_type=self.args.mix_type)
                mix_pooled_rep = mix_output_dict['normalized_pooled_rep']
                mix_output = mix_output_dict['y']
                mix_masks = mix_output_dict['masks']

                if 'til' in self.args.scenario:
                    mix_output = mix_output[t]

                n_loss,_=self.hat_criterion_adapter(mix_output,targets,mix_masks) # it self is also training
                amix_loss+=n_loss # let's first do some pre-training

                if self.args.amix_head:
                    mix_pooled_reps.append(mix_output.unsqueeze(1).clone())
                else:
                    mix_pooled_reps.append(mix_pooled_rep.unsqueeze(1).clone())

        elif self.args.attn_type == 'self':
            orders = self.order_generation(t)
            print('orders: ',orders)

            neg_mix_pooled_reps = []

            for order_id,order in enumerate(orders):
                mix_output_dict = self.model(t,input_ids, segment_ids, input_mask,s=s,start_mixup=True,l=order,idx=order_id,mix_type=self.args.mix_type)
                mix_output = mix_output_dict['y']
                mix_masks = mix_output_dict['masks']
                mix_pooled_rep = mix_output_dict['normalized_pooled_rep']

                if 'til' in self.args.scenario:
                    mix_output = mix_output[t]
                n_loss,_=self.hat_criterion_adapter(mix_output,targets,mix_masks) # it self is also training
                amix_loss+=n_loss # let's first do some pre-training

                if self.args.use_dissimilar:
                    neg_mix_pooled_rep = mix_output_dict['neg_normalized_pooled_rep']
                    neg_mix_pooled_reps.append(neg_mix_pooled_rep.unsqueeze(1).clone())

                if self.args.amix_head:
                    mix_pooled_reps.append(mix_output.unsqueeze(1).clone())
                else:
                    mix_pooled_reps.append(mix_pooled_rep.unsqueeze(1).clone())

            if self.args.use_dissimilar:
                neg_mix_pooled_reps.append(neg_mix_pooled_rep.unsqueeze(1).clone()) # in fact, this is only use as negative samples
                neg_mix_pooled_reps = torch.cat(neg_mix_pooled_reps, dim=1)

        cur_mix_outputs = torch.cat(mix_pooled_reps, dim=1)

        if self.args.use_dissimilar: #dissimilar as negative
            cur_mix_outputs = torch.cat([cur_mix_outputs,neg_mix_pooled_reps],dim=0)
            targets_ =  torch.cat([targets,targets],dim=0)
            amix_loss += self.sup_con(cur_mix_outputs, targets_,args=self.args)

        else:
            amix_loss += self.sup_con(cur_mix_outputs, targets,args=self.args) #train attention and contrastive learning at the same time
        return amix_loss


    def order_generation(self,t):
        orders = []
        nsamples = t
        for n in range(self.args.naug):
            if n == 0: orders.append([pre_t for pre_t in range(t)])
            elif nsamples>=1:
                orders.append(random.Random(self.args.seed).sample([pre_t for pre_t in range(t)],nsamples))
                nsamples-=1
        return orders

    def idx_generator(self,bsz):
        #TODO: why don't we generate more?
        ls,idxs = [],[]
        for n in range(self.args.ntmix):
            if self.args.tmix:
                if self.args.co:
                    mix_ = np.random.choice([0, 1], 1)[0]
                else:
                    mix_ = 1

                if mix_ == 1:
                    l = np.random.beta(self.args.alpha, self.args.alpha)
                    if self.args.separate_mix:
                        l = l
                    else:
                        l = max(l, 1-l)
                else:
                    l = 1
                idx = torch.randperm(bsz) # Note I currently do not havce unsupervised data
            ls.append(l)
            idxs.append(idx)

        return idxs,ls


    def hat_criterion_adapter_aux(self,outputs,targets,masks,t=None):
        reg=0
        count=0
        ewc_loss=0

        if self.aux_mask_pre is not None:
            for key in set(masks.keys()) & set(self.aux_mask_pre.keys()):
                m = masks[key]
                mp = self.aux_mask_pre[key]
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m_key,m_value in masks.items():
                reg+=m_value.sum()
                count+=np.prod(m_value.size()).item()
        reg/=count


        return self.ce(outputs,targets)+self.lamb*reg,reg


    def hat_criterion_adapter(self,outputs,targets,masks):
        reg=0
        count=0

        if self.mask_pre is not None:
            # for m,mp in zip(masks,self.mask_pre):
            for key in set(masks.keys()) & set(self.mask_pre.keys()):
                m = masks[key]
                mp = self.mask_pre[key]
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m_key,m_value in masks.items():
                reg+=m_value.sum()
                count+=np.prod(m_value.size()).item()
        reg/=count

        return self.ce(outputs,targets)+self.lamb*reg,reg


    def f1_compute_fn(self,y_true, y_pred,average):
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return f1_score(y_true, y_pred,average=average)


    def acc_compute_fn(self,y_true, y_pred):
        try:
            from sklearn.metrics import accuracy_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return accuracy_score(y_true, y_pred)
