import sys,time
import numpy as np
import torch
from tqdm import tqdm, trange
from copy import deepcopy

import utils
import sys
sys.path.append("./approaches/base/")
from cnn_base import Appr as ApprBase
import torch.nn.functional as F

# adapt from https://github.com/joansj/hat/blob/master/src/approaches/sgd.py


class Appr(ApprBase):

    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)


        print('CNN ONE')

        return


    def train(self,t,train,valid,num_train_steps,train_data,valid_data): #N-CL
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
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc))
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
                t.to(self.device) if t is not None else None for t in batch]
            images,targets= batch

            # Forward
            output_dict=self.model.forward(images)
            outputs = output_dict['y']
            pooled_rep = output_dict['normalized_pooled_rep']

            output=outputs[t] #TIL setting

            # print('targets: ',targets)

            loss=self.ce(output,targets)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())


            if self.args.sup_loss:
                if self.args.sup_head_norm:
                    output_rep = F.normalize(output, dim=1)
                    loss += self.sup_loss(output_rep,pooled_rep,images,targets)
                else:
                    loss += self.sup_loss(output,pooled_rep,images,targets)

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
                images,targets= batch
                real_b=targets.size(0)

                output_dict = self.model.forward(images)
                outputs = output_dict['y']

                output=outputs[t] #TIL
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
