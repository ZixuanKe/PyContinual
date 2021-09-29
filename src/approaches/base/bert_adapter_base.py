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
from seqeval.metrics import classification_report
import torch.nn.functional as F
import nlp_data_utils as data_utils
from pytorch_pretrained_bert.optimization import BertAdam
from copy import deepcopy
sys.path.append("./approaches/")
from contrastive_loss import SupConLoss, CRDLoss
from buffer import Buffer as Buffer


class Appr(object):

    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x


    def __init__(self,model,logger,taskcla, args=None):

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
        self.sup_con = SupConLoss(temperature=args.temp,base_temperature=args.base_temp)

        # shared ==============
        self.model=model
        self.model_old=None
        self.train_batch_size=args.train_batch_size
        self.eval_batch_size=args.eval_batch_size
        self.args=args
        self.ce=torch.nn.CrossEntropyLoss()
        self.taskcla = taskcla
        self.logger = logger

        if 'ewc' in args.approach:
            self.lamb=args.lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000
            self.fisher=None

        #OWM ============
        if 'owm' in args.approach:
            dtype = torch.cuda.FloatTensor  # run on GPU
            self.P1 = torch.autograd.Variable(torch.eye(self.args.bert_adapter_size).type(dtype), volatile=True) #inference only
            self.P2 = torch.autograd.Variable(torch.eye(self.args.bert_adapter_size).type(dtype), volatile=True)

        #UCL ======================
        if 'ucl' in args.approach:
            self.saved = 0
            self.beta = args.beta
            self.model=model
            self.model_old = deepcopy(self.model)

        if 'one' in args.approach:
            self.model=model
            self.initial_model=deepcopy(model)

        if 'der' in args.approach:
            self.buffer = Buffer(self.args.buffer_size, self.device)
            self.mse = torch.nn.MSELoss()

        if 'gem' in args.approach:
            self.buffer = Buffer(self.args.buffer_size, self.device)
            # Allocate temporary synaptic memory
            self.grad_dims = []
            for pp in model.parameters():
                self.grad_dims.append(pp.data.numel())

            self.grads_cs = []
            self.grads_da = torch.zeros(np.sum(self.grad_dims)).to(self.device)

        if 'a-gem' in args.approach:
            self.buffer = Buffer(self.args.buffer_size, self.device)
            self.grad_dims = []
            for param in self.model.parameters():
                self.grad_dims.append(param.data.numel())
            self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
            self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

        if 'l2' in args.approach:
            self.lamb=self.args.lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
            self.regularization_terms = {}
            self.task_count = 0
            self.online_reg = False  # True: There will be only one importance matrix and previous model parameters
                                    # False: Each task has its own importance matrix and model parameters
        print('BERT ADAPTER BASE')

        return

    def augment_current_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets, t):
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

            pre_output_dict = self.model(input_ids, segment_ids, input_mask,l=l,idx=idx,start_mixup=True,mix_type='tmix') #pre_t as indicator
            pre_pooled_rep = pre_output_dict['normalized_pooled_rep']
            pre_output = pre_output_dict['y']
            pre_output = pre_output[t]
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


    def sup_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets,t):
        if self.args.sup_head:
            outputs = torch.cat([output.clone().unsqueeze(1), output.clone().unsqueeze(1)], dim=1)
        else:
            outputs = torch.cat([pooled_rep.clone().unsqueeze(1), pooled_rep.clone().unsqueeze(1)], dim=1)

        loss = self.sup_con(outputs, targets,args=self.args)
        return loss


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


    def f1_compute_fn(self,y_true, y_pred,average):
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return f1_score(y_true, y_pred,average=average)

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

        return self.ce(output,targets)+self.lamb*loss_reg

