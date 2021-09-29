import sys,time
import numpy as np
import torch
from tqdm import tqdm, trange
import sys
sys.path.append("./approaches/base/")
from cnn_base import Appr as ApprBase
import utils
import torch.nn.functional as F
from torch.utils.data import DataLoader


# adapt from https://github.com/joansj/hat/blob/master/src/approaches/sgd.py


class Appr(ApprBase):

    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        print('CNN DERPP NCL')


        return



    def train(self,t,train,valid,num_train_steps,train_data,valid_data): #N-CL
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

        samples_per_task = int(len(train_data) * self.args.buffer_percent)
        loader = DataLoader(train_data, batch_size=samples_per_task)

        images,targets = next(iter(loader))
        images = images.to(self.device)
        targets = targets.to(self.device)

        output_dict = self.model.forward(images)
        if 'dil' in self.args.scenario:
            cur_task_output=output_dict['y']
        elif 'til' in self.args.scenario:
            outputs=output_dict['y']
            cur_task_output = outputs[t]


        self.buffer.add_data(
            examples=images,
            labels=targets,
            task_labels=torch.ones(samples_per_task,dtype=torch.long).to(self.device) * (t),
            logits = cur_task_output.data ,
            segment_ids=images, #dumy
            input_mask=images,
        )

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
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]

            loss=self.ce(output,targets)

            if not self.buffer.is_empty():
                buf_inputs, buf_labels,buf_logits, buf_task_labels, buf_segment_ids,buf_input_mask = self.buffer.get_data(
                    self.args.buffer_size)
                buf_inputs = buf_inputs
                buf_labels = buf_labels.long()
                buf_logits = buf_logits

                output_dict = self.model(buf_inputs)
                if 'dil' in self.args.scenario:
                    cur_task_output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    cur_task_output = outputs[t]

                loss += self.args.beta * self.ce(cur_task_output, buf_labels)
                loss += self.args.alpha * self.mse(cur_task_output, buf_logits)


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
                images,targets= batch
                real_b=targets.size(0)

                output_dict = self.model.forward(images)

                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs = output_dict['y']
                    if self.args.ent_id: #detected id
                        output_d= self.ent_id_detection(trained_task,images,t=t)
                        output = output_d['output']
                    else:
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

