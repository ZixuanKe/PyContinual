import sys,time
import numpy as np
import torch
from tqdm import tqdm, trange
import sys
sys.path.append("./approaches/base/")
from cnn_base import Appr as ApprBase
import utils
import torch.nn.functional as F
from copy import deepcopy


class Appr(ApprBase):
    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        print('CNN MTL')


        return


    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train(self,t,train,valid,num_train_steps,train_data,valid_data): #N-CL
        self.model=deepcopy(self.initial_model) # Restart model

        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                self.train_epoch(t,train,iter_bar)
                clock1=time.time()
                train_loss=self.eval_validation(t,train)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f} |'.format(e+1,
                    1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss),end='')

                valid_loss=self.eval_validation(t,valid)
                print(' Valid: loss={:.3f} |'.format(valid_loss),end='')
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
        except KeyboardInterrupt:
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
            images,targets,tasks= batch

            # Forward
            output_dict=self.model.forward(images)
            outputs = output_dict['y']

            if 'til' in self.args.scenario:
                output = self.til_output_friendly(tasks,outputs)
            elif 'dil' in self.args.scenario:
                output=outputs

            loss=self.criterion_train(tasks,output,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval_validation(self,_,data):
        total_loss=0
        total_num=0
        self.model.eval()

        target_list = []
        pred_list = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    t.to(self.device) if t is not None else None for t in batch]
                images,targets,tasks= batch
                real_b=targets.size(0)

                # Forward
                output_dict=self.model.forward(images)
                outputs = output_dict['y']

                if 'til' in self.args.scenario:
                    output = self.til_output_friendly(tasks,outputs)
                elif 'dil' in self.args.scenario:
                    output=outputs
                loss=self.criterion_train(tasks,output,targets)

                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_num+=real_b

        return total_loss/total_num

    def eval(self,t,data,test=None,trained_task=None):
        # This is used for the test. All tasks separately
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
                images,targets,tasks= batch
                real_b=targets.size(0)

                # Forward
                output_dict=self.model.forward(images)
                outputs = output_dict['y']

                if 'til' in self.args.scenario:
                    output = self.til_output_friendly(tasks,outputs)
                elif 'dil' in self.args.scenario:
                    output=outputs


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


    def criterion_train(self,tasks,output,targets):
        #MTL training
        loss=self.ce(output,targets)*len(tasks)
        return loss/targets.size(0)


    # def criterion_train(self,tasks,outputs,targets):
    #     loss=0
    #     for t in np.unique(tasks.data.cpu().numpy()):
    #         t=int(t)
    #         output=outputs #always shared head
    #         idx=(tasks==t).data.nonzero().view(-1)
    #         loss+=self.criterion(output[idx,:],targets[idx])*len(idx)
    #     return loss/targets.size(0)
    #
