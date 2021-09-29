import sys,time
import numpy as np
import torch
# from copy import deepcopy

import utils
from tqdm import tqdm, trange
import math
sys.path.append("./approaches/base/")
from w2v_cnn_base import Appr as ApprBase
rnn_weights = [
    'krn.lstm.rnn.weight_ih_l0',
    'krn.lstm.rnn.weight_hh_l0',
    'krn.lstm.rnn.bias_ih_l0',
    'krn.lstm.rnn.bias_hh_l0',
    'krn.gru.rnn.weight_ih_l0',
    'krn.gru.rnn.weight_hh_l0',
    'krn.gru.rnn.bias_ih_l0',
    'krn.gru.rnn.bias_hh_l0',
    'krn.gru.gru_cell.x2h.weight',
    'krn.gru.gru_cell.x2h.bias',
    'krn.gru.gru_cell.h2h.weight',
    'krn.gru.gru_cell.h2h.bias'
]

class Appr(ApprBase):
    def __init__(self,model,args=None,taskcla=None,logger=None):
    # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('W2V + RNN SRK NCL')

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        if self.args.optimizer == 'sgd':
            print('sgd')
            return torch.optim.SGD(self.model.parameters(),lr=lr)
        elif self.args.optimizer == 'adam':
            print('adam')
            return torch.optim.Adam(self.model.parameters(),lr=lr)

    def train(self,t,train,valid,num_train_steps,train_data,valid_data):
        # self.model=deepcopy(self.initial_model) # Restart model: isolate

        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            self.train_epoch(t,train,iter_bar)
            clock1=time.time()
            train_loss,train_acc,train_f1_macro=self.eval(t,train)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
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
                    self.optimizer=self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model,best_model)

        return



    def train_epoch(self,t,data,iter_bar):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            tokens_term_ids, tokens_sentence_ids, targets= batch
            s = (1-0.1) * math.exp(-0.1*t)+0.1
            # print('s: ',s) samller than 1

            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)

            # Forward
            # outputs,fln_outputs,krn_outputs,krn_hidden=self.model.forward(task,input_ids, segment_ids, input_mask)
            output_dict = self.model.forward(task,tokens_term_ids, tokens_sentence_ids)
            control_1=output_dict['control_1']
            control_2=output_dict['control_2']
            control_3=output_dict['control_3']

            if 'dil' in self.args.scenario:
                output = output_dict['y']
                fln_output=output_dict['fln_y']
                krn_output=output_dict['krn_y']

            elif 'til' in self.args.scenario:
                outputs= output_dict['y']
                fln_outputs=output_dict['fln_y']
                krn_outputs=output_dict['krn_y']

                output=outputs[t]
                fln_output=fln_outputs[t]
                krn_output=krn_outputs[t]

            loss=self.ce(output,targets) + self.ce(fln_output,targets) + self.ce(krn_output,targets)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            self.control_1_s += torch.abs(control_1.sum(0))
            self.control_1_s = (self.control_1_s - torch.min(self.control_1_s))/(torch.max(self.control_1_s)-torch.min(self.control_1_s))
            control_1_sorted, control_1_indices = torch.sort(self.control_1_s)

            control_1_mask = torch.zeros_like(self.control_1_s).cuda()
            control_1_mask[:int(self.control_1_s.size(-1)*s)] = 1 - control_1_sorted[:int(self.control_1_s.size(-1)*s)]

            self.control_2_s += torch.abs(control_2.sum(0))
            self.control_2_s = (self.control_2_s - torch.min(self.control_2_s))/(torch.max(self.control_2_s)-torch.min(self.control_2_s))
            control_2_sorted, control_2_indices = torch.sort(self.control_2_s)

            control_2_mask = torch.zeros_like(self.control_2_s).cuda()
            control_2_mask[:int(self.control_2_s.size(-1)*s)] = 1 - control_2_sorted[:int(self.control_2_s.size(-1)*s)]

            self.control_3_s += torch.abs(control_3.sum(0))
            self.control_3_s = (self.control_3_s - torch.min(self.control_3_s))/(torch.max(self.control_3_s)-torch.min(self.control_3_s))
            control_3_sorted, control_3_indices = torch.sort(self.control_3_s)

            control_3_mask = torch.zeros_like(self.control_3_s).cuda()
            control_3_mask[:int(self.control_3_s.size(-1)*s)] = 1 - control_3_sorted[:int(self.control_3_s.size(-1)*s)]



            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            if t>0:
                for n,p in self.model.named_parameters():
                    if n in rnn_weights:
                        p.grad.data*=self.model.get_view_for(n,control_1_mask,control_2_mask,control_3_mask)



            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

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
                    bat.to(self.device) if bat is not None else None for bat in batch]
                tokens_term_ids, tokens_sentence_ids, targets= batch
                real_b=tokens_term_ids.size(0)
                task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)
                # outputs,fln_outputs,krn_outputs,krn_hidden = self.model.forward(task,input_ids, segment_ids, input_mask)

                output_dict = self.model.forward(task,tokens_term_ids, tokens_sentence_ids)
                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                    fln_output=output_dict['fln_y']
                    krn_output=output_dict['krn_y']

                elif 'til' in self.args.scenario:
                    outputs = output_dict['y']
                    fln_outputs=output_dict['fln_y']
                    krn_outputs=output_dict['krn_y']
                    output=outputs[t]
                    fln_output=fln_outputs[t]
                    krn_output=krn_outputs[t]

                loss=self.ce(output,targets) + self.ce(fln_output,targets) + self.ce(krn_output,targets)

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

