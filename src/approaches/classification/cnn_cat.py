import sys,time
import numpy as np
import torch
from tqdm import tqdm, trange

import utils
from copy import deepcopy
sys.path.append("./approaches/base/")
from cnn_base import Appr as ApprBase



#TODO: adapt the CAT here. please make the code cleaner, this will later made public

########################################################################################################################

class Appr(ApprBase):

    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        return


    def auto_similarity(self,t,train,valid,num_train_steps,train_data,valid_data):
        #TODO: to detect the task similarity

        for pre_task in range(t+1):

            if 'mlp' in self.args.approach:
                gfc1,gfc2 = self.model.mask(pre_task)
                pre_mask=[gfc1.detach(),gfc2.detach()]
            elif 'cnn' in self.args.approach:
                gc1,gc2,gc3,gfc1,gfc2 = self.model.mask(pre_task)
                pre_mask=[gc1.detach(),gc2.detach(),gc3.detach(),gfc1.detach(),gfc2.detach()]

            if pre_task == t: # the last one
                print('>>> Now Training Phase: {:6s} <<<'.format('reference'))
                self.real_train(t,train,valid,num_train_steps,train_data,valid_data,phase='reference',
                                            pre_mask=pre_mask,pre_task=pre_task) # implemented as random mask
            elif pre_task != t:
                print('>>> Now Training Phase: {:6s} <<<'.format('transfer'))
                self.real_train(t,train,valid,num_train_steps,train_data,valid_data,phase='transfer',
                                            pre_mask=pre_mask,pre_task=pre_task)

            if pre_task == t: # the last one
                test_loss,test_acc=self.eval_(t,valid,phase='reference',
                                             pre_mask=pre_mask,pre_task=pre_task)
            elif pre_task != t:
                test_loss,test_acc=self.eval_(t,valid,phase='transfer',
                                             pre_mask=pre_mask,pre_task=pre_task)

            self.acc_transfer[t,pre_task]=test_acc
            self.lss_transfer[t,pre_task]=test_loss

        # print('test_acc: ',self.acc_transfer[t][:t+1])
        # print('test_loss: ',self.lss_transfer[t][:t+1])
        print('Save at transfer_acc')
        np.savetxt(self.args.output + '_acc_transfer',self.acc_transfer,'%.4f',delimiter='\t')

        similarity = [0]
        if t > 0:
            acc_list = self.acc_transfer[t][:t] #t from 0
            print('acc_list: ',acc_list)

            similarity = [0 if (acc_list[acc_id] <= self.acc_transfer[t][t]) else 1 for acc_id in range(len(acc_list))] # remove all acc < 0.5

            for source_task in range(len(similarity)):
                self.similarity_transfer[t,source_task]=similarity[source_task]

        print('Save at similarity_transfer')
        np.savetxt(self.args.output + '_similarity_transfer',self.similarity_transfer,'%.4f',delimiter='\t')

        print('similarity: ',similarity)
        return similarity


    def train(self,t,train,valid,num_train_steps,train_data,valid_data): #N-CL
        similarity = self.auto_similarity(t,train,valid,num_train_steps,train_data,valid_data)
        self.similarities.append(similarity)
        print('similarity: ',self.similarities[-1])
        print('similarities: ',self.similarities)

        self.check_federated.set_similarities(self.similarities)
        self.real_train(t,train,valid,num_train_steps,train_data,valid_data)



    def real_train(self,t,train,valid,num_train_steps,train_data,valid_data,phase='mcl',pre_task=None,pre_mask=None):

        #TODO: before the real training, we defenitely need to first detect the task similarity



        self.model.transfer=deepcopy(self.transfer_initial_model) # Restart transfer network: isolate

        best_loss=np.inf
        best_model=utils.get_model(self.model)

        print('phase: ',phase)

        if phase=='mcl' or phase=='transfer' or phase=='reference':
            lr=self.lr
            patience=self.lr_patience
            nepochs=self.nepochs


        self.optimizer=self._get_optimizer_cat(lr,phase)

        try:
            for e in range(nepochs):
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                self.train_epoch(t,train,iter_bar,phase=phase,pre_mask=pre_mask,
                                 pre_task=pre_task)
                clock1=time.time()
                train_loss,train_acc=self.eval_(t,train,trained_task=t,phase=phase,pre_mask=pre_mask,
                                               pre_task=pre_task)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc))
                # Valid
                valid_loss,valid_acc=self.eval_(t,valid,trained_task=t,phase=phase,pre_mask=pre_mask,
                                               pre_task=pre_task)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
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
                        self.optimizer=self._get_optimizer_cat(lr,phase)
                print()
        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model,best_model)


        if phase=='mcl':
            # Activations mask
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            mask=self.model.mask(task,s=self.smax)


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

        #TODO: make the end function separately
        if phase=='mcl':
            self.history_mask_pre.append([m.data.clone() for m in self.mask_pre])


        return

    def train_epoch(self,t,data,iter_bar,phase=None, pre_mask=None, pre_task=None):
        self.model.train()

        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                t.to(self.device) if t is not None else None for t in batch]
            images,targets= batch

            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax

            # Forward

            if phase == 'mcl':
                output_dict=self.model(t,images,s=s,phase=phase,similarity=self.similarities[-1],
                                                      history_mask_pre=self.history_mask_pre,check_federated=self.check_federated)

                masks = output_dict['masks']

                if 'dil' in self.args.scenario:
                    output_attn = output_dict['y_attn']
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs_attn = output_dict['y_attn']
                    outputs = output_dict['y']

                    output_attn = outputs_attn[t]
                    output = outputs[t]


                if output_attn is None:
                    loss=self.criterion(output,targets,masks)
                else:
                    loss=self.joint_criterion(output,targets,masks,output_attn)


            elif phase == 'transfer' or phase == 'reference':

                output_dict=self.model(t,images,s=s,phase=phase,pre_mask=pre_mask, pre_task=pre_task,
                                   history_mask_pre=self.history_mask_pre,check_federated=self.check_federated)

                if 'dil' in self.args.scenario:
                    output = output_dict['y'] # for TIL, this will need more change
                elif 'til' in self.args.scenario:
                    outputs = output_dict['y']
                    output = outputs[t]

                loss=self.transfer_criterion(output,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()



            if phase == 'mcl':
                # Restrict layer gradients in backprop
                if t>0:
                    for n,p in self.model.named_parameters():
                        if n in self.mask_back and p.grad is not None:
                            Tsim_mask=self.model.Tsim_mask(t,history_mask_pre=self.history_mask_pre,similarity=self.similarities[-1])
                            Tsim_vals=self.model.get_view_for(n,Tsim_mask).clone()
                            p.grad.data*=torch.max(self.mask_back[n],Tsim_vals)


                # Compensate embedding gradients
                for n,p in self.model.named_parameters():
                    if n.startswith('mcl.e') and p.grad is not None:
                        num=torch.cosh(torch.clamp(s*p.data,-self.args.thres_cosh,self.args.thres_cosh))+1
                        den=torch.cosh(p.data)+1
                        p.grad.data*=self.smax/s*num/den



            elif phase == 'reference':
                # Compensate embedding gradients
                for n,p in self.model.named_parameters():
                    if n.startswith('transfer.e')  and p.grad is not None:
                        num=torch.cosh(torch.clamp(s*p.data,-self.args.thres_cosh,self.args.thres_cosh))+1
                        den=torch.cosh(p.data)+1
                        p.grad.data*=self.smax/s*num/den

            # Apply step
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            if phase == 'mcl':
                # Constrain embeddings
                for n,p in self.model.named_parameters():
                    if n.startswith('mcl.e'):
                        p.data=torch.clamp(p.data,-self.args.thres_emb,self.args.thres_emb)


            elif phase == 'reference':
                # Constrain embeddings
                for n,p in self.model.named_parameters():
                    if n.startswith('transfer.e'):
                        p.data=torch.clamp(p.data,-self.args.thres_emb,self.args.thres_emb)
        return



    #TODO: I don't see how data loader is useful here
    def eval_(self,t,test,trained_task=None,phase=None,pre_mask=None, pre_task=None):
        total_mask_acc,total_mask_loss,total_att_acc,total_att_loss,total_num,f1_mask,f1_att = \
            self.compute_acc(t,test,trained_task,phase,pre_mask,pre_task)

        if 'all-one' in self.args.similarity_detection:
            total_loss = total_att_loss
            total_acc = total_att_acc

        if phase=='mcl' and 'no_attention' not in self.args.loss_type:
            if total_att_acc > total_mask_acc:
                total_loss = total_att_loss
                total_acc = total_att_acc
            else:
                total_loss = total_mask_loss
                total_acc = total_mask_acc

        else:
                total_loss = total_mask_loss
                total_acc = total_mask_acc

        return total_loss/total_num,total_acc/total_num


    def eval(self,t,test,valid,trained_task=None,phase=None):

        choose_att = False
        total_mask_acc,total_mask_loss,total_att_acc,total_att_loss,total_num,f1_mask,f1_att = \
            self.compute_acc(t,valid,trained_task,phase)

        if 'all-one' in self.args.similarity_detection:
            choose_att = True
        elif phase=='mcl' and 'no_attention' not in self.args.loss_type:
            if total_att_acc > total_mask_acc:
                choose_att = True

        print('choose_att: ',choose_att)
        #Here simply use validation to choose attention in testing.
        # One can also remember which tasks have used the attention in training and then apply attention for testing

        total_mask_acc,total_mask_loss,total_att_acc,total_att_loss,total_num,f1_mask,f1_att = \
            self.compute_acc(t,test,trained_task,phase)

        if choose_att == True:
            total_loss = total_att_loss
            total_acc = total_att_acc
            total_f1 = f1_att
        else:
            total_loss = total_mask_loss
            total_acc = total_mask_acc
            total_f1 = f1_mask

        return total_loss/total_num,total_acc/total_num,total_f1


    def compute_acc(self,t,data,trained_task=None,phase=None,pre_mask=None,pre_task=None):
        total_att_loss=0
        total_att_acc=0

        total_mask_loss=0
        total_mask_acc=0

        total_num=0
        self.model.eval()
        target_list = []
        pred_att_list = []
        pred_mask_list = []

        print('phase: ',phase)

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    t.to(self.device) if t is not None else None for t in batch]
                images,targets= batch
                real_b=targets.size(0)
                target_list.append(targets)

                # Forward

                if phase == 'mcl':


                    if 'dil' in self.args.scenario:
                        if self.args.last_id: # fix 0
                            output_dict =self.model(trained_task,images,s=self.smax,phase=phase,similarity=self.similarities[-1],
                                                    history_mask_pre=self.history_mask_pre,check_federated=self.check_federated)
                            masks = output_dict['masks']
                            output_attn = output_dict['y_attn']
                            output = output_dict['y'] # for TIL, this will need more change

                        if self.args.ent_id: #detected id
                            output_d= self.ent_id_detection(trained_task,images,t=t)
                            output = output_d['output']
                            masks = output_d['masks']
                            output_attn = output_d['output_attn']

                    elif 'til' in self.args.scenario:

                        if self.args.ent_id: #detected id
                            output_d= self.ent_id_detection(trained_task,images,t=t)
                            output = output_d['output']
                            masks = output_d['masks']
                            output_attn = output_d['output_attn']

                        else:
                            output_dict = self.model.forward(t,images,s=self.smax,phase=phase,similarity=self.similarities[-1],
                                                        history_mask_pre=self.history_mask_pre,check_federated=self.check_federated)
                            outputs_attn = output_dict['y_attn']
                            outputs = output_dict['y']

                            output_attn = outputs_attn[t]
                            output = outputs[t]
                            masks = output_dict['masks']

                    if output_attn is None:
                        loss=self.criterion(output,targets,masks)
                    else:
                        loss=self.joint_criterion(output,targets,masks,output_attn)

                elif phase == 'transfer' or phase == 'reference':
                    output_dict=self.model(t,images,s=self.smax,phase=phase,pre_mask=pre_mask, pre_task=pre_task)

                    if 'dil' in self.args.scenario:
                        output = output_dict['y'] # for TIL, this will need more change
                    elif 'til' in self.args.scenario:
                        outputs = output_dict['y']
                        output = outputs[t]

                    loss=self.transfer_criterion(output,targets)


                # if phase=='mcl' and (similarity is not None and t<len(similarity) and np.count_nonzero(similarity[:t])>1 and similarity[t]==1):

                if phase=='mcl' and 'no_attention' not in self.args.loss_type and output_attn is not None:
                    _,att_pred=output_attn.max(1)
                    _,mask_pred=output.max(1)

                    pred_att_list.append(att_pred)
                    pred_mask_list.append(mask_pred)

                    mask_hits=(mask_pred==targets).float()
                    att_hits=(att_pred==targets).float()

                    # Log
                    total_mask_loss+=loss.data.cpu().numpy().item()*real_b
                    total_mask_acc+=mask_hits.sum().data.cpu().numpy().item()

                    # Log
                    total_att_loss+=loss.data.cpu().numpy().item()*real_b
                    total_att_acc+=att_hits.sum().data.cpu().numpy().item()


                else:
                    _,pred=output.max(1)
                    hits=(pred==targets).float()
                    pred_mask_list.append(pred)

                    # Log
                    total_mask_loss+=loss.data.cpu().numpy().item()*real_b
                    total_mask_acc+=hits.sum().data.cpu().numpy().item()


                total_num+=real_b

            f1_mask=self.f1_compute_fn(y_pred=torch.cat(pred_mask_list,0),y_true=torch.cat(target_list,0),average='macro')
            if len(pred_att_list) > 1:
                f1_att=self.f1_compute_fn(y_pred=torch.cat(pred_att_list,0),y_true=torch.cat(target_list,0),average='macro')
            else:
                f1_att = None
        return total_mask_acc,total_mask_loss,total_att_acc,total_att_loss,total_num,f1_mask,f1_att

    def transfer_criterion(self,outputs,targets,mask=None):
        return self.ce(outputs,targets)


    def joint_criterion(self,outputs,targets,masks,outputs_attn):
        return self.criterion(outputs,targets,masks) + self.args.model_weights*self.ce(outputs_attn,targets)

    def criterion(self,outputs,targets,masks):
        reg=0
        count=0

        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count


        return self.ce(outputs,targets)+self.lamb*reg





########################################################################################################################
