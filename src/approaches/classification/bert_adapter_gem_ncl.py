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
from seqeval.metrics import classification_report
import torch.nn.functional as F
import nlp_data_utils as data_utils
from copy import deepcopy
from torch.utils.data import DataLoader
import quadprog
sys.path.append("./approaches/base/")
from bert_adapter_base import Appr as ApprBase
from my_optimization import BertAdam

class Appr(ApprBase):


    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('BERT ADAPTER GEM NCL')

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
            global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step)
            clock1=time.time()

            train_loss,train_acc,train_f1_macro=self.eval(t,train)
            clock2=time.time()
            # print('time: ',float((clock1-clock0)*10*25))
            self.logger.info('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc))
            # print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
            #     1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')

            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
            self.logger.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc))
            # print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')

            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                self.logger.info(' *')
                # print(' *',end='')
            print()
            # break
        # Restore best
        utils.set_model_(self.model,best_model)



        # Update old
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights

        self.grads_cs.append(torch.zeros(
            np.sum(self.grad_dims)).to(self.device))

        # add data to the buffer
        print('len(train): ',len(train_data))
        samples_per_task = int(len(train_data) * self.args.buffer_percent)
        print('samples_per_task: ',samples_per_task)

        loader = DataLoader(train_data, batch_size=samples_per_task)

        input_ids, segment_ids, input_mask, targets,_ = next(iter(loader))

        self.buffer.add_data(
            examples=input_ids.to(self.device),
            segment_ids=segment_ids.to(self.device),
            input_mask=input_mask.to(self.device),
            labels=targets.to(self.device),
            task_labels=torch.ones(samples_per_task,
                dtype=torch.long).to(self.device) * (t)
        )

        return

    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step):
        self.num_labels = self.taskcla[t][1]
        self.model.train()
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _= batch


            # Forward current model

            if not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_task_labels, buf_segment_ids,buf_input_mask = self.buffer.get_data(
                    self.args.buffer_size)
                # print('buf_segment_ids: ',buf_segment_ids.size())
                # print('buf_input_mask: ',buf_input_mask.size())

                for tt in buf_task_labels.unique():
                    # compute gradient on the memory buffer
                    optimizer.zero_grad()
                    cur_task_inputs = buf_inputs[buf_task_labels == tt].long()
                    cur_task_segment = buf_segment_ids[buf_task_labels == tt].long()
                    cur_task_mask = buf_input_mask[buf_task_labels == tt].long()
                    cur_task_labels = buf_labels[buf_task_labels == tt].long()
                    output_dict = self.model.forward(cur_task_inputs, cur_task_segment, cur_task_mask)
                    if 'dil' in self.args.scenario:
                        cur_task_output=output_dict['y']
                    elif 'til' in self.args.scenario:
                        outputs=output_dict['y']
                        cur_task_output = outputs[tt]

                    penalty = self.ce(cur_task_output, cur_task_labels)
                    penalty.backward()
                    self.store_grad(self.model.parameters, self.grads_cs[tt], self.grad_dims)


            optimizer.zero_grad()
            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            # Forward
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]


            loss=self.ce(output,targets)

            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            loss.backward()

            # check if gradient violates buffer constraints
            if not self.buffer.is_empty():
                # copy gradient
                self.store_grad(self.model.parameters, self.grads_da, self.grad_dims)

                dot_prod = torch.mm(self.grads_da.unsqueeze(0),
                                torch.stack(self.grads_cs).T)
                if (dot_prod < 0).sum() != 0:
                    self.project2cone2(self.grads_da.unsqueeze(1),
                                  torch.stack(self.grads_cs).T, margin=self.args.gamma)
                    # copy gradients back
                    self.overwrite_grad(self.model.parameters, self.grads_da,
                                   self.grad_dims)


            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        return global_step

    def eval(self,t,data,test=None,trained_task=None):
        total_loss=0
        total_acc=0
        total_num=0
        target_list = []
        pred_list = []


        with torch.no_grad():
            self.model.eval()

            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets, _= batch
                real_b=input_ids.size(0)

                output_dict = self.model.forward(input_ids, segment_ids, input_mask)
                # Forward
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]


                loss=self.ce(output,targets)

                _,pred=output.max(1)
                hits=(pred==targets).float()

                target_list.append(targets)
                pred_list.append(pred)

                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b

            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')

                # break

        return total_loss/total_num,total_acc/total_num,f1




    def store_grad(self,params, grads, grad_dims):
        """
            This stores parameter gradients of past tasks.
            pp: parameters
            grads: gradients
            grad_dims: list with number of parameters per layers
        """
        # store the gradients
        grads.fill_(0.0)
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = np.sum(grad_dims[:count + 1])
                grads[begin: end].copy_(param.grad.data.view(-1))
            count += 1


    def overwrite_grad(self,params, newgrad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = sum(grad_dims[:count + 1])
                this_grad = newgrad[begin: end].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            count += 1


    def project2cone2(self,gradient, memories, margin=0.5, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.
            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
        """
        memories_np = memories.cpu().t().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        n_rows = memories_np.shape[0]
        self_prod = np.dot(memories_np, memories_np.transpose())
        self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
        grad_prod = np.dot(memories_np, gradient_np) * -1
        G = np.eye(n_rows)
        h = np.zeros(n_rows) + margin
        v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.from_numpy(x).view(-1, 1))
