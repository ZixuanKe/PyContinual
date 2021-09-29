import sys,time
import numpy as np
import torch
# from copy import deepcopy

import utils
from tqdm import tqdm, trange
from torch.nn import functional as F
from torch.utils.data import DataLoader

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


        # Update old

        # add data to the buffer
        print('len(train): ',len(train_data))
        samples_per_task = int(len(train_data) * self.args.buffer_percent)
        print('samples_per_task: ',samples_per_task)

        loader = DataLoader(train_data, batch_size=samples_per_task)
        tokens_term_ids, tokens_sentence_ids, targets = next(iter(loader))

        input_ids = tokens_term_ids.to(self.device)
        segment_ids = tokens_sentence_ids.to(self.device)
        input_mask = tokens_term_ids.to(self.device) #dummpy
        targets = targets.to(self.device)


        output_dict = self.model.forward(input_ids, segment_ids)
        if 'dil' in self.args.scenario:
            cur_task_output=output_dict['y']
        elif 'til' in self.args.scenario:
            outputs=output_dict['y']
            cur_task_output = outputs[t]

        self.buffer.add_data(
            examples=input_ids,
            segment_ids=segment_ids,
            input_mask=input_mask,
            labels=targets,
            task_labels=torch.ones(samples_per_task,dtype=torch.long).to(self.device) * (t),
            logits = cur_task_output.data
        )


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
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]


            loss=self.ce(output,targets)



            if not self.buffer.is_empty():
                buf_inputs, buf_labels,buf_logits, buf_task_labels, buf_segment_ids,buf_input_mask = self.buffer.get_data(
                    self.args.buffer_size)

                buf_task_inputs = buf_inputs.long()
                buf_task_segment = buf_segment_ids.long()
                buf_task_mask = buf_input_mask.long() #dummy
                buf_task_labels = buf_labels.long()
                buf_task_logits = buf_logits

                output_dict = self.model.forward(buf_task_inputs, buf_task_segment)
                if 'dil' in self.args.scenario:
                    cur_task_output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    cur_task_output = outputs[t]

                loss += self.args.beta * self.ce(cur_task_output, buf_task_labels)
                loss += self.args.alpha * self.mse(cur_task_output, buf_task_logits)


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

