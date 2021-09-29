import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils
from tqdm import tqdm, trange
sys.path.append("./approaches/base/")
from w2v_cnn_base import Appr as ApprBase

class Appr(ApprBase):
    # def __init__(self,model,nepochs=200,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        print('W2V ONE')

        return

    # def train(self,t,train,valid,args):
    def train(self,t,train,valid,num_train_steps,train_data,valid_data):

        self.model=deepcopy(self.initial_model) # Restart model: isolate


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

            outputs=output_dict['y']
            output=outputs[t]
            loss=self.ce(output,targets)


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
                outputs=output_dict['y']

                output=outputs[t]
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
