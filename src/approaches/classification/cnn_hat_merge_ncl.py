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

    def __init__(self,model,aux_model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,aux_model=aux_model,logger=logger,taskcla=taskcla,args=args)
        #aux for HAT
        #model for NCL

        print('DIL CNN HAT AUX NCL')

        return


    def train(self,t,train,valid,num_train_steps,train_data,valid_data): #N-CL

        best_loss=np.inf
        best_aux_model=utils.get_model(self.aux_model)
        best_model=utils.get_model(self.model)

        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer_merge(lr)
        print('self.nepochs: ',self.nepochs)
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                self.train_epoch(t,train,iter_bar,e=e)
                clock1=time.time()
                train_loss,train_acc,train_f1_macro=self.eval(t,train)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc))
                # Valid
                valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_aux_model=utils.get_model(self.aux_model)
                    best_model=utils.get_model(self.model)

                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer_merge(lr)
                print()
        except KeyboardInterrupt:
            print()
        
        # Restore best validation model
        utils.set_model_(self.aux_model,best_aux_model)
        utils.set_model_(self.model,best_model)

        # Activations mask
        mask=self.aux_model.mask(t,s=self.smax)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        if t==0:
            self.mask_pre=mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])
        
        # Weights mask
        self.mask_back={}
        for n,_ in self.aux_model.named_parameters():
            vals=self.aux_model.get_view_for(n,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals

        if self.args.weight_align:
            if t >= 0:
                self.model.save_old_norm()
            if t > 0:
                self.model.align_norms()

        return


    def train_epoch(self,t,data,iter_bar,e):
        self.aux_model.train()
        self.model.train()

        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                t.to(self.device) if t is not None else None for t in batch]
            images,targets= batch

            # BASE NET ===============
            output_dict=self.model(images)
            pooled_rep = output_dict['normalized_pooled_rep']

            if 'dil' in self.args.scenario:
                output = output_dict['y']
            elif 'til' in self.args.scenario:
                outputs = output_dict['y']
                output = outputs[t]


            loss=self.args.ce_ratio * self.ce(output,targets)
            #
            # AUX NET ===============
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax

            aux_output_dict = self.aux_model(t,images,s=s)
            aux_masks = aux_output_dict['masks']

            if 'dil' in self.args.scenario:
                aux_output = aux_output_dict['y']
            elif 'til' in self.args.scenario:
                aux_outputs = output_dict['y']
                aux_output = aux_outputs[t]



            aux_loss,_=self.criterion_hat(aux_output,targets,aux_masks) #output_ce is the output of head (no softmax)
            loss += aux_loss


            if self.args.amix and t > 0: # Do we want this
                if self.args.amix_head_norm:
                    print('amix_head_norm')
                    output_rep = F.normalize(output, dim=1)
                    loss += self.args.amix_ratio * self.amix_loss(output_rep,pooled_rep,images,targets, t,s,use_aux=True)
                else:
                    loss += self.args.amix_ratio *self.amix_loss(output,pooled_rep,images,targets, t,s,use_aux=True)

            if self.args.augment_distill and t > 0: #separatlu append
                if self.args.distill_head_norm:
                    print('distill_head_norm')
                    output_rep = F.normalize(output, dim=1)
                    loss += self.augment_distill_loss(output_rep,pooled_rep,images,targets, t,use_aux=True)
                else:
                    loss += self.augment_distill_loss(output,pooled_rep,images,targets, t,use_aux=True)

            if self.args.sup_loss:
                if self.args.sup_head_norm:
                    print('sup_head_norm')
                    output_rep = F.normalize(output, dim=1)
                    loss += self.args.sup_ratio *self.sup_loss(output_rep,pooled_rep,images,targets)
                else:
                    loss += self.args.sup_ratio *self.sup_loss(output,pooled_rep,images,targets)


            # # supervised and transfer contrastive loss ===============
            #
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            #
            # # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # # # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.aux_model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.aux_model.named_parameters():
                if n.startswith('e'):
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den
            #
            # # Apply step
            torch.nn.utils.clip_grad_norm(self.aux_model.parameters(),self.clipgrad)
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

            self.optimizer.step()

            # Constrain embeddings
            for n,p in self.aux_model.named_parameters():
                if n.startswith('e'):
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)




        return


    def eval(self,t,data,test=None,trained_task=None):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        self.aux_model.eval()

        target_list = []
        pred_list = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    t.to(self.device) if t is not None else None for t in batch]
                images,targets= batch
                real_b=targets.size(0)

                output_dict = self.model(images)

                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    if self.args.ent_id: #detected id
                        output_d= self.ent_id_detection(trained_task,images,t=t)
                        output = output_d['output']
                    else:
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




########################################################################################################################
