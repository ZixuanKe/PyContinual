import time
import numpy as np
import torch
from tqdm import tqdm, trange
import utils
import torch.nn as nn
from copy import deepcopy
import functools
import torch.nn.functional as F
import sys
sys.path.append("./approaches/base/")
from cnn_base import Appr as ApprBase

########################################################################################################################
# adapt from https://github.com/joansj/hat/blob/master/src/approaches/hat.py



#TODO: CNN based contrastive learning
class Appr(ApprBase):

    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        print('DIL CNN AEM NCL')

        return


    def train(self,t,train,valid,num_train_steps,train_data,valid_data): #N-CL
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                self.train_epoch(t,train,iter_bar,e=e)
                clock1=time.time()
                train_loss,train_acc,train_f1_macro=self.eval(t,train,trained_task=t)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc))
                # Valid
                valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid,trained_task=t)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')
                elif  e > self.args.e_start_mixup:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            # break # no break for now
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer(lr)
                print()
        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model,best_model)

        # Activations mask
        mask=self.model.mask(t,s=self.smax)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        if t==0:
            self.mask_pre=mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

        # Weights mask
        self.mask_back={}
        for n,_ in self.model.named_parameters():
            vals=self.model.get_view_for(n,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals


        return

    def train_epoch(self,t,data,iter_bar,e):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                t.to(self.device) if t is not None else None for t in batch]
            images,targets= batch

            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax

            # supervised CE loss ===============
            output_dict = self.model(t,images,s=s)
            masks = output_dict['masks']

            output = output_dict['y']

            loss,_=self.criterion_hat(output,targets,masks)

            if self.args.contrastive_by_confident:
                if self.args.no_defer: contrastive_idx = 1 #no deffered for contrastive
                else: contrastive_idx = e // int(self.args.nepochs * self.args.contrastive_defer_rate)  # 80%

                if contrastive_idx > 0:
                    contrastive_loss_weighted_sum = self.my_con(
                        output_dict['weighted_sum'],
                        output_dict['order_outputs'],
                        output_dict['weight'])

                    loss += self.args.contrastive_weight*contrastive_loss_weighted_sum

            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den

            # Apply step
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)


        return

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
                if self.args.distill_loss:
                    images,targets,idx,contrastive_idx= batch
                else:
                    images,targets= batch
                real_b=targets.size(0)

                if self.args.mix_id:
                    output_dict=self.model.mix_output(trained_task,images,s=self.smax)
                    output = output_dict['y']
                    masks = output_dict['masks']

                elif self.args.last_id: # fix 0
                    output_dict=self.model.forward(trained_task,images,s=self.smax)
                    output = output_dict['y']
                    masks = output_dict['masks']

                elif self.args.true_id: # fix 0
                    output_dict=self.model.forward(t,images,s=self.smax)
                    output = output_dict['y']
                    masks = output_dict['masks']

                elif self.args.ent_id:
                    output_d= self.ent_id_detection(trained_task,images,t=t)
                    masks = output_d['masks']
                    output = output_d['output']


                # Forward
                loss,reg=self.criterion_hat(output,targets,masks)
                # loss=self.ce(output,targets)

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




########################################################################################################################
