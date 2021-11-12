import sys, time
import numpy as np
import torch

dtype = torch.cuda.FloatTensor  # run on GPU
import utils
from tqdm import tqdm, trange
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
import os
import logging
import glob
import math
import json
import random
sys.path.append("./approaches/base/")
from bert_adapter_base import Appr as ApprBase
from my_optimization import BertAdam

########################################################################################################################

class Appr(ApprBase):


    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('BERT ADAPTER OWM NCL')

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
            global_step=self.train_epoch(e,t,train,iter_bar, optimizer,t_total,global_step)
            clock1=time.time()

            train_loss,train_acc,train_f1_macro=self.eval(t,train)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')

            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                print(' *',end='')

            print()
            # break
        # Restore best
        utils.set_model_(self.model,best_model)

        return

    def train_epoch(self,cur_epoch,t,data,iter_bar,optimizer,t_total,global_step):
        self.num_labels = self.taskcla[t][1]
        self.model.train()
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _= batch

            # Forward
            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            x_list=output_dict['x_list']
            h_list=output_dict['h_list']
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]
            loss = self.ce(output, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            lamda = step / len(batch)/self.args.num_train_epochs + cur_epoch/self.args.num_train_epochs

            alpha_array = [1.0 * 0.00001 ** lamda, 1.0 * 0.0001 ** lamda, 1.0 * 0.01 ** lamda, 1.0 * 0.1 ** lamda]

            def pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):
                x=x.detach()
                p=p.detach()

                if cnn:
                    _, _, H, W = x.shape
                    F, _, HH, WW = w.shape
                    S = stride  # stride
                    Ho = int(1 + (H - HH) / S)
                    Wo = int(1 + (W - WW) / S)
                    for i in range(Ho):
                        for j in range(Wo):
                            # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                            r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                            # r = r[:, range(r.shape[1] - 1, -1, -1)]
                            k = torch.mm(p, torch.t(r))
                            p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
                else:
                    r = x
                    k = torch.mm(p, torch.t(r))
                    p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data, torch.t(p.data))
            # Compensate embedding gradients
            for n, w in self.model.named_parameters():
                for layer_id in range(self.args.bert_num_hidden_layers):
                    # if n == 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_owm.c1.weight': #
                    #     pro_weight(self.Pc1, x_list[0], w, alpha=alpha_array[0], stride=2)
                    #
                    # if n == 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_owm.c2.weight':
                    #     pro_weight(self.Pc2, x_list[1], w, alpha=alpha_array[0], stride=2)
                    #
                    # if n == 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_owm.c3.weight':
                    #     pro_weight(self.Pc3, x_list[2], w, alpha=alpha_array[0], stride=2)

                    if n == 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_owm.fc1.weight':
                        # print('h_list: ',len(h_list))
                        pro_weight(self.P1,  h_list[layer_id][0], w, alpha=alpha_array[1], cnn=False)

                    if n == 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_owm.fc2.weight':
                        pro_weight(self.P2,  h_list[layer_id][1], w, alpha=alpha_array[2], cnn=False)


            # Apply step
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        return global_step

    def eval(self,t,data,test=None,trained_task=None):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        target_list = []
        pred_list = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets, _= batch
                real_b=input_ids.size(0)
                # Forward

                # Forward
                output_dict = self.model.forward(input_ids, segment_ids, input_mask)
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]
                loss = self.ce(output, targets)
                _, pred = output.max(1)
                hits = (pred % 10 == targets).float()
                target_list.append(targets)
                pred_list.append(pred)
                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b
            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')

        return total_loss / total_num, total_acc / total_num,f1

