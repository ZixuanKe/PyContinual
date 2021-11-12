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
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
import utils
# from apex import amp

import torch.nn.functional as F
import functools
import torch.nn as nn
from copy import deepcopy
sys.path.append("./approaches/base/")
from bert_adapter_mask_base import Appr as ApprBase
from my_optimization import BertAdam




class Appr(ApprBase):


    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('DIL BERT ADAPTER MASK SUP NCL')

        return

    def train(self,t,train,valid,num_train_steps,train_data,valid_data):

        global_step = 0
        self.model.to(self.device)

        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=t_total)


        best_loss=np.inf
        best_model=utils.get_model(self.model)

        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step,e)
            clock1=time.time()

            train_loss,train_acc,train_f1_macro=self.eval(t,train,trained_task=t)
            clock2=time.time()
            self.logger.info('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc))

            # print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
            #     1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')

            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid,trained_task=t)
            self.logger.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc))

            # print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                print(' *',end='')

            print()
            # break
        # Restore best
        utils.set_model_(self.model,best_model)

        # Activations mask
        # task=torch.autograd.Variable(torch.LongTensor([t]).to(self.device),volatile=False)

        if self.args.multi_gpu and not self.args.distributed:
            mask=self.model.module.mask(t,s=self.smax)
        else:
            mask=self.model.mask(t,s=self.smax)
        for key,value in mask.items():
            mask[key]=torch.autograd.Variable(value.data.clone(),requires_grad=False)
        if t==0:
            self.mask_pre=mask
        else:
            for key,value in self.mask_pre.items():
                self.mask_pre[key]=torch.max(self.mask_pre[key],mask[key])

        # Weights mask
        self.mask_back={}
        for n,p in self.model.named_parameters():
            if self.args.multi_gpu and not self.args.distributed:
                vals=self.model.module.get_view_for(n,p,self.mask_pre)
            else:
                vals=self.model.get_view_for(n,p,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals


        return

    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step,e):

        self.model.train()
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _= batch
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax

            # supervised CE loss ===============
            output_dict = self.model(t,input_ids, segment_ids, input_mask,s=s)
            masks = output_dict['masks']
            pooled_rep = output_dict['normalized_pooled_rep']
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]


            loss,_=self.hat_criterion_adapter(output,targets,masks) #output_ce is the output of head (no softmax)


            # transfer contrastive ===============

            if self.args.amix and t > 0:
                loss += self.amix_loss(output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s)

            if self.args.augment_distill and t > 0: #separatlu append
                loss += self.augment_distill_loss(output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s)

            if self.args.sup_loss:
                loss += self.sup_loss(output,pooled_rep,input_ids, segment_ids, input_mask,targets,t,s)


            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            loss.backward()

            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.model.named_parameters():
                if 'adapter_mask.e' in n or n.startswith('e'):
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den

            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step

            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if 'adapter_mask.e' in n or n.startswith('e'):
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

            # break
        return global_step





    def eval(self,t,data,test=None,trained_task=None):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        target_list = []
        pred_list = []

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets, _= batch
                real_b=input_ids.size(0)

                if 'dil' in self.args.scenario:

                    if self.args.last_id: # use the last one
                        output_dict = self.model(trained_task,input_ids, segment_ids, input_mask,s=self.smax)
                        output = output_dict['y']
                        masks = output_dict['masks']

                    elif self.args.ent_id: # detect the testing is
                        outputs = []
                        entropies = []

                        if trained_task is None: #training
                            entrop_to_test = range(0, t + 1)
                        else: #testing
                            entrop_to_test = range(0, trained_task + 1)

                        for e in entrop_to_test:
                            output_dict = self.model(e,input_ids, segment_ids, input_mask,s=self.smax)
                            output = output_dict['y']
                            masks = output_dict['masks']
                            outputs.append(output) #shared head

                            Y_hat = F.softmax(output, -1)
                            entropy = -1*torch.sum(Y_hat * torch.log(Y_hat))
                            entropies.append(entropy)
                        inf_task_id = torch.argmin(torch.stack(entropies))
                        # self.logger.info('inf_task_id: '+str(inf_task_id))
                        output=outputs[inf_task_id] #but don't know which one

                elif 'til' in self.args.scenario:
                    task=torch.LongTensor([t]).cuda()
                    output_dict=self.model.forward(task,input_ids, segment_ids, input_mask,s=self.smax)
                    outputs = output_dict['y']
                    masks = output_dict['masks']
                    output = outputs[t]


                loss,_=self.hat_criterion_adapter(output,targets,masks)

                _,pred=output.max(1)
                hits=(pred==targets).float()
                target_list.append(targets)
                pred_list.append(pred)
                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b

            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')


        return total_loss/total_num,total_acc/total_num,f1




