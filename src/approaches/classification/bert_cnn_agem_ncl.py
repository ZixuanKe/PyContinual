import sys,time
import numpy as np
import torch
from copy import deepcopy
import quadprog
import utils
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
sys.path.append("./approaches/base/")
from bert_cnn_base import Appr as ApprBase
#TODO: GEM is very expensive, consider A-GEM

class Appr(ApprBase):
    """ A-GEM adpted from https://github.com/aimagelab/mammoth/blob/master/models/agem.py """

    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)


        print('CONTEXTUAL CNN AGEM NCL')

        return



    def train(self,t,train,valid,num_train_steps,train_data,valid_data):
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

            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.args.train_batch_size*(clock1-clock0)/len(train),1000*self.args.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')
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

        # Update old

        # add data to the buffer
        print('len(train): ',len(train_data))
        samples_per_task = int(len(train_data) * self.args.buffer_percent)
        print('samples_per_task: ',samples_per_task)

        loader = DataLoader(train_data, batch_size=samples_per_task)

        input_ids, segment_ids, input_mask, targets,_ = next(iter(loader))

        self.buffer.add_data(
            examples=input_ids.to(self.device),
            segment_ids=segment_ids.to(self.device),
            input_mask=input_mask.to(self.device),
            labels=targets.to(self.device),
            task_labels=torch.ones(samples_per_task,
                dtype=torch.long).to(self.device) * (t)
        )

        return

    def train_epoch(self,t,data,iter_bar):
        self.model.train()

        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets,_= batch


            # now compute the grad on the current data
            self.optimizer.zero_grad()
            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]

            loss = self.ce(output, targets)
            loss.backward() #backward first


            # Forward current model
            if not self.buffer.is_empty():
                self.store_grad(self.model.parameters, self.grad_xy, self.grad_dims)
                buf_inputs, buf_labels, buf_task_labels, buf_segment_ids,buf_input_mask = self.buffer.get_data(
                    self.args.buffer_size)
                self.model.zero_grad()
                buf_inputs = buf_inputs.long()
                buf_segment = buf_segment_ids.long()
                buf_mask = buf_input_mask.long()
                buf_labels = buf_labels.long()

                output_dict = self.model.forward(buf_inputs, buf_segment, buf_mask)
                if 'dil' in self.args.scenario:
                    cur_output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    cur_output = outputs[t]

                penalty = self.ce(cur_output, buf_labels)
                penalty.backward()
                self.store_grad(self.model.parameters, self.grad_er, self.grad_dims)

                dot_prod = torch.dot(self.grad_xy, self.grad_er)
                if dot_prod.item() < 0:
                    g_tilde = self.project(gxy=self.grad_xy, ger=self.grad_er)
                    self.overwrite_grad(self.model.parameters, g_tilde, self.grad_dims)
                else:
                     self.overwrite_grad(self.model.parameters, self.grad_xy, self.grad_dims)




            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

            # Backward
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
                input_ids, segment_ids, input_mask, targets,_= batch
                real_b=input_ids.size(0)

                # Forward
                output_dict = self.model.forward(input_ids, segment_ids, input_mask)

                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
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


