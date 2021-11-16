# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys, time, os
import numpy as np
import torch
import copy
import utils

from copy import deepcopy
from tqdm import tqdm
sys.path.append('../')
from networks.classification.discriminator import Discriminator
sys.path.append("./approaches/base/")
from cnn_base import Appr as ApprBase
from torch.utils.data import DataLoader

#TODO: consider whether adapt it to our framwork or use their framework
#https://github.com/facebookresearch/Adversarial-Continual-Learning/issues?q=is%3Aissue+is%3Aclosed
class Appr(ApprBase):

    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        print('CNN ACL NCL')



        # optimizer & adaptive lr
        self.e_lr=[args.e_lr] * args.ntasks
        self.d_lr=[args.d_lr] * args.ntasks

        self.lr_min=args.lr_min
        self.lr_factor=args.lr_factor
        self.lr_patience=args.lr_patience


        self.adv_loss_reg=args.adv
        self.diff_loss_reg=args.orth
        self.s_steps=args.s_step
        self.d_steps=args.d_step

        self.diff=args.diff


        # Initialize generator and discriminator
        self.model=model
        self.discriminator=self.get_discriminator(0)
        self.discriminator.get_size()

        if 'mlp' in args.approach:
            self.latent_dim=args.mlp_adapter_size #if nlp
        else:
             self.latent_dim=2048 #if nlp

        self.task_loss=torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_d=torch.nn.CrossEntropyLoss().to(self.device)
        self.adversarial_loss_s=torch.nn.CrossEntropyLoss().to(self.device)
        self.diff_loss=DiffLoss().to(self.device)

        self.optimizer_S=self.get_S_optimizer(0)
        self.optimizer_D=self.get_D_optimizer(0)

        self.task_encoded={}

        self.mu=0.0
        self.sigma=1.0

        print()

    def get_discriminator(self, t):
        discriminator=Discriminator(self.args, t).to(self.device)
        return discriminator

    def get_S_optimizer(self, t, e_lr=None):
        if e_lr is None: e_lr=self.e_lr[t]
        optimizer_S=torch.optim.SGD(self.model.parameters(), momentum=self.args.mom,
                                    weight_decay=self.args.e_wd, lr=e_lr)
        return optimizer_S

    def get_D_optimizer(self, t, d_lr=None):
        if d_lr is None: d_lr=self.d_lr[t]
        optimizer_D=torch.optim.SGD(self.discriminator.parameters(), weight_decay=self.args.d_wd, lr=d_lr)
        return optimizer_D

    def train(self, t,train,valid,num_train_steps,train_data,valid_data):
        self.discriminator=self.get_discriminator(t)

        best_loss=np.inf
        best_model=utils.get_model(self.model)


        best_loss_d=np.inf
        best_model_d=utils.get_model(self.discriminator)

        dis_lr_update=True
        d_lr=self.d_lr[t]
        patience_d=self.lr_patience
        self.optimizer_D=self.get_D_optimizer(t, d_lr)

        e_lr=self.e_lr[t]
        patience=self.lr_patience
        self.optimizer_S=self.get_S_optimizer(t, e_lr)


        for e in range(self.nepochs):

            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')

            self.train_epoch(t,train,iter_bar)
            clock1=time.time()

            train_res=self.eval_(t,train)

            utils.report_tr(train_res, e, self.args.train_batch_size, clock0, clock1)

            # Valid
            valid_res=self.eval_(t,valid)
            utils.report_val(valid_res)

            # Adapt lr for S and D
            if valid_res['loss_tot'] < best_loss:
                best_loss=valid_res['loss_tot']
                best_model=utils.get_model(self.model)
                patience=self.lr_patience
                print(' *', end='')
            else:
                patience-=1
                if patience <= 0:
                    e_lr/=self.lr_factor
                    print(' lr={:.1e}'.format(e_lr), end='')
                    if e_lr < self.lr_min:
                        print()
                        break
                    patience=self.lr_patience
                    self.optimizer_S=self.get_S_optimizer(t, e_lr)

            if train_res['loss_a'] < best_loss_d:
                best_loss_d=train_res['loss_a']
                best_model_d=utils.get_model(self.discriminator)
                patience_d=self.lr_patience
            else:
                patience_d-=1
                if patience_d <= 0 and dis_lr_update:
                    d_lr/=self.lr_factor
                    print(' Dis lr={:.1e}'.format(d_lr))
                    if d_lr < self.lr_min:
                        dis_lr_update=False
                        print("Dis lr reached minimum value")
                        print()
                    patience_d=self.lr_patience
                    self.optimizer_D=self.get_D_optimizer(t, d_lr)
            print()

        # Restore best validation model (early-stopping)
        self.model.load_state_dict(copy.deepcopy(best_model))
        self.discriminator.load_state_dict(copy.deepcopy(best_model_d))


        samples_per_task = int(len(train_data) * self.args.buffer_percent)
        loader = DataLoader(train_data, batch_size=samples_per_task)

        images,targets = next(iter(loader))
        images = images.to(self.device)
        targets = targets.to(self.device)


        self.buffer.add_data(
            examples=images,
            labels=targets,
            task_labels=torch.ones(samples_per_task,dtype=torch.long).to(self.device) * (t),
            segment_ids=images, #dumy
            input_mask=images,
        )


    def train_epoch(self, t,data,iter_bar):

        self.model.train()
        self.discriminator.train()

        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            images,targets= batch
            bsz = images.size(0)
            x=images.to(device=self.device)
            y=targets.to(device=self.device, dtype=torch.long)
            task = torch.LongTensor([t]).repeat(bsz) #TODO: tt should have batch size large
            tt=task.to(device=self.device)
            td=tt+1


            if not self.buffer.is_empty(): # ACL does use memeory
                buf_inputs, buf_labels, buf_task_labels, buf_segment_ids,buf_input_mask = self.buffer.get_data(
                    self.args.buffer_size)
                buf_inputs = buf_inputs
                buf_labels = buf_labels.long()

                x = torch.cat([x,buf_inputs])
                y = torch.cat([y,buf_labels])
                tt = torch.cat([tt,buf_task_labels])
                td = torch.cat([td,buf_task_labels+1])

            # Detaching samples in the batch which do not belong to the current task before feeding them to P
            t_current=t * torch.ones_like(tt)
            body_mask=torch.eq(t_current, tt).cpu().numpy()
            # x_task_module=data.to(device=self.device)
            x_task_module=x.clone()
            for index in range(x.size(0)):
                if body_mask[index] == 0:
                    x_task_module[index]=x_task_module[index].detach()
            x_task_module=x_task_module.to(device=self.device)

            # Discriminator's real and fake task labels
            t_real_D=td.to(self.device) #td = tt+1
            t_fake_D=torch.zeros_like(t_real_D).to(self.device)

            # ================================================================== #
            #                        Train Shared Module                          #
            # ================================================================== #
            # training S for s_steps
            for s_step in range(self.s_steps):
                self.optimizer_S.zero_grad()
                self.model.zero_grad()

                output_dict=self.model(x, x_task_module, tt, t)
                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs = output_dict['y']
                    output = outputs[t]

                task_loss=self.task_loss(output, y)

                shared_encoded, task_encoded=self.model.get_encoded_ftrs(x, x_task_module, t)
                dis_out_gen_training=self.discriminator.forward(shared_encoded, t_real_D, t)
                adv_loss=self.adversarial_loss_s(dis_out_gen_training, t_real_D)

                if self.diff == 'yes':
                    diff_loss=self.diff_loss(shared_encoded, task_encoded)
                else:
                    diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.diff_loss_reg=0

                total_loss=task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss
                total_loss.backward(retain_graph=True)

                self.optimizer_S.step()

            # ================================================================== #
            #                          Train Discriminator                       #
            # ================================================================== #
            # training discriminator for d_steps
            for d_step in range(self.d_steps):
                self.optimizer_D.zero_grad()
                self.discriminator.zero_grad()

                # training discriminator on real data
                output_dict=self.model(x, x_task_module, tt, t)
                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs = output_dict['y']
                    output = outputs[t]

                shared_encoded, task_out=self.model.get_encoded_ftrs(x, x_task_module, t)
                dis_real_out=self.discriminator.forward(shared_encoded.detach(), t_real_D, t)
                dis_real_loss=self.adversarial_loss_d(dis_real_out, t_real_D)
                if self.args.experiment == 'miniimagenet':
                    dis_real_loss*=self.adv_loss_reg
                dis_real_loss.backward(retain_graph=True)

                # training discriminator on fake data
                z_fake=torch.as_tensor(np.random.normal(self.mu, self.sigma, (x.size(0), self.latent_dim)),dtype=torch.float32, device=self.device)
                dis_fake_out=self.discriminator.forward(z_fake, t_real_D, t)
                dis_fake_loss=self.adversarial_loss_d(dis_fake_out, t_fake_D)
                if self.args.experiment == 'miniimagenet':
                    dis_fake_loss*=self.adv_loss_reg
                dis_fake_loss.backward(retain_graph=True)

                self.optimizer_D.step()

        return


    def eval_(self, t,data):
        loss_a, loss_t, loss_d, loss_total=0, 0, 0, 0
        correct_d, correct_t = 0, 0
        num=0

        self.model.eval()
        self.discriminator.eval()

        res={}
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    t.to(self.device) if t is not None else None for t in batch]
                images,targets= batch
                bsz = images.size(0)
                x=images.to(device=self.device)
                y=targets.to(device=self.device, dtype=torch.long)
                task = torch.LongTensor([t]).repeat(bsz)
                tt=task.to(device=self.device)
                t_real_D=tt+1

                # Forward
                output_dict=self.model(x, x, tt, t)

                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs = output_dict['y']
                    output = outputs[t]

                shared_out, task_out=self.model.get_encoded_ftrs(x, x, t)
                _, pred=output.max(1)
                correct_t+=pred.eq(y.view_as(pred)).sum().item()

                # Discriminator's performance:
                output_d=self.discriminator.forward(shared_out, t_real_D, t)
                _, pred_d=output_d.max(1)
                correct_d+=pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

                # Loss values
                task_loss=self.task_loss(output, y)
                adv_loss=self.adversarial_loss_d(output_d, t_real_D)

                if self.diff == 'yes':
                    diff_loss=self.diff_loss(shared_out, task_out)
                else:
                    diff_loss=torch.tensor(0).to(device=self.device, dtype=torch.float32)
                    self.diff_loss_reg=0

                total_loss = task_loss + self.adv_loss_reg * adv_loss + self.diff_loss_reg * diff_loss

                loss_t+=task_loss
                loss_a+=adv_loss
                loss_d+=diff_loss
                loss_total+=total_loss

                num+=x.size(0)

        res['loss_t'], res['acc_t']=loss_t.item() / (step + 1), 100 * correct_t / num
        res['loss_a'], res['acc_d']=loss_a.item() / (step + 1), 100 * correct_d / num
        res['loss_d']=loss_d.item() / (step + 1)
        res['loss_tot']=loss_total.item() / (step + 1)
        res['size']=self.loader_size(data)

        return res

    #

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
                    t.to(self.device) if t is not None else None for t in batch]
                images,targets= batch
                real_b=targets.size(0)
                x=images.to(device=self.device)


                if 'dil' in self.args.scenario:
                    if self.args.last_id: # fix 0
                        task = torch.LongTensor([trained_task]).repeat(real_b)
                        tt=task.to(device=self.device)
                        output_dict=self.model(x, x, tt, trained_task)
                        output = output_dict['y']

                    elif self.args.ent_id:
                        output_d= self.ent_id_detection(trained_task,images,t=t)
                        output = output_d['output']

                elif 'til' in self.args.scenario:
                    if self.args.ent_id: #detected id
                        output_d= self.ent_id_detection(trained_task,images,t=t)
                        output = output_d['output']
                    else:
                        task = torch.LongTensor([t]).repeat(real_b)
                        tt=task.to(device=self.device)
                        output_dict = self.model.forward(x, x, tt, t)
                        outputs = output_dict['y']
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


        return total_loss/total_num,total_acc/total_num,f1


    def loader_size(self, data_loader):
        return data_loader.dataset.__len__()

        #
class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
