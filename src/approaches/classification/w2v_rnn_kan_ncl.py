import sys,time
import numpy as np
import torch
# from copy import deepcopy

import utils
from tqdm import tqdm, trange
import torch.nn.functional as F
sys.path.append("./approaches/base/")
from w2v_cnn_base import Appr as ApprBase

rnn_weights = [
    'mcl.lstm.rnn.weight_ih_l0',
    'mcl.lstm.rnn.weight_hh_l0',
    'mcl.lstm.rnn.bias_ih_l0',
    'mcl.lstm.rnn.bias_hh_l0',
    'mcl.gru.rnn.weight_ih_l0',
    'mcl.gru.rnn.weight_hh_l0',
    'mcl.gru.rnn.bias_ih_l0',
    'mcl.gru.rnn.bias_hh_l0']

class Appr(ApprBase):
    def __init__(self,model,args=None,taskcla=None,logger=None):
    # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        print('W2V + RNN NCL')

        return



    def train(self,t,train,valid,num_train_steps,train_data,valid_data):
        # self.model=deepcopy(self.initial_model) # Restart model: isolate


        if t == 0: which_types = ['mcl']
        else: which_types = ['ac','mcl']

        for which_type in which_types:

            print('Training Type: ',which_type)

            best_loss=np.inf
            best_model=utils.get_model(self.model)
            lr=self.lr
            patience=self.lr_patience
            self.optimizer=self._get_optimizer_kan(lr,which_type)

            # Loop epochs
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                self.train_epoch(t,train,iter_bar,which_type)
                clock1=time.time()
                train_loss,train_acc,train_f1_macro=self.eval(t,train,train_data,which_type,trained_task=t)
                clock2=time.time()
                print('time: ',float((clock1-clock0)*10*25))

                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid,valid_data,which_type,trained_task=t)
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
                        self.optimizer=self._get_optimizer_kan(lr,which_type)
                print()

            # Restore best
            utils.set_model_(self.model,best_model)

        return



    def train_epoch(self,t,data,iter_bar,which_type):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            tokens_term_ids, tokens_sentence_ids, targets= batch
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)

            # Forward
            output_dict=self.model.forward(task,tokens_term_ids, tokens_sentence_ids,which_type,s)

            if 'dil' in self.args.scenario:
                output = output_dict['y']
            elif 'til' in self.args.scenario:
                outputs = output_dict['y']
                output = outputs[t]

            loss=self.ce(output,targets) #no regularization
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            if t>0 and which_type=='mcl':
                task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
                mask=self.model.ac.mask(task,s=self.smax)
                mask = torch.autograd.Variable(mask.data.clone(),requires_grad=False)
                for n,p in self.model.named_parameters():
                    if n in rnn_weights:
                        # print('n: ',n)
                        # print('p: ',p.grad.size())
                        p.grad.data*=self.model.get_view_for(n,mask)

            # Compensate embedding gradients
            for n,p in self.model.ac.named_parameters():
                if 'ac.e' in n:
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den


            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            # Constrain embeddings
            for n,p in self.model.ac.named_parameters():
                if 'ac.e' in n:
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

        return

    def eval(self,t,data,test,which_type,trained_task):
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
                tokens_term_ids, tokens_sentence_ids, targets= batch
                real_b=tokens_term_ids.size(0)


                if 'dil' in self.args.scenario:

                    if self.args.last_id: # fix 0
                        task=torch.LongTensor([trained_task]).cuda()
                        output_dict=self.model.forward(task,tokens_term_ids, tokens_sentence_ids,which_type,s=self.smax)
                        output = output_dict['y']
                    elif self.args.ent_id:
                        output_d= self.ent_id_detection(trained_task,tokens_term_ids, tokens_sentence_ids,t,which_type)
                        output = output_d['output']

                elif 'til' in self.args.scenario:
                    task=torch.LongTensor([t]).cuda()
                    output_dict=self.model.forward(task,tokens_term_ids, tokens_sentence_ids,which_type,s=self.smax)
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

