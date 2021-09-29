import sys,time
import numpy as np
import torch
# from copy import deepcopy

import utils
from tqdm import tqdm, trange
sys.path.append("./approaches/base/")
from w2v_cnn_base import Appr as ApprBase

class Appr(ApprBase):

    # def __init__(self,model,nepochs=200,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('W2V NCL')
        return

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
            # print('time: ',float((clock1-clock0)*30*25))

            # print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
            #     1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')

            self.logger.info('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc))

            # Valid
            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
            # print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            self.logger.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc))


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

        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance()

        # Save the weight and importance of weights of current task
        self.task_count += 1
        if self.online_reg and len(self.regularization_terms)>0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {'importance':importance, 'task_param':task_param}
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {'importance':importance, 'task_param':task_param}

        return



    def train_epoch(self,t,data,iter_bar):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                t.to(self.device) if t is not None else None for t in batch]
            tokens_term_ids, tokens_sentence_ids, targets= batch
            # print('tokens_term_ids: ',tokens_term_ids)
            # Forward
            output_dict=self.model.forward(tokens_term_ids, tokens_sentence_ids)
            pooled_rep = output_dict['normalized_pooled_rep']
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]


            loss=self.criterion(output,targets)

            if self.args.sup_loss:
                loss += self.sup_loss(output,pooled_rep,tokens_term_ids, tokens_sentence_ids,targets,t)


            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
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
                    t.to(self.device) if t is not None else None for t in batch]
                tokens_term_ids, tokens_sentence_ids, targets= batch
                real_b=tokens_term_ids.size(0)

                output_dict = self.model.forward(tokens_term_ids, tokens_sentence_ids)
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]

                loss=self.criterion(output,targets)

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

    def calculate_importance(self):
        # Use an identity importance so it is an L2 regularization.
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)  # Identity
        return importance


    def criterion(self, inputs, targets, regularization=True, **kwargs):
        loss =self.ce(inputs, targets, **kwargs)

        if regularization and len(self.regularization_terms)>0:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0
            for i,reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            loss += self.lamb * reg_loss
        return loss