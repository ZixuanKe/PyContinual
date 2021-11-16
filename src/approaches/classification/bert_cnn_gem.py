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
    """ GEM adpted from https://github.com/aimagelab/mammoth/blob/master/models/gem.py """

    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)


        print('CONTEXTUAL CNN EWC NCL')

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
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights

        self.grads_cs.append(torch.zeros(
            np.sum(self.grad_dims)).to(self.device))

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

            # Forward current model

            if not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_task_labels, buf_segment_ids,buf_input_mask = self.buffer.get_data(
                    self.args.buffer_size)
                # print('buf_segment_ids: ',buf_segment_ids.size())
                # print('buf_input_mask: ',buf_input_mask.size())

                for tt in buf_task_labels.unique():
                    # compute gradient on the memory buffer
                    self.optimizer.zero_grad()
                    cur_task_inputs = buf_inputs[buf_task_labels == tt].long()
                    cur_task_segment = buf_segment_ids[buf_task_labels == tt].long()
                    cur_task_mask = buf_input_mask[buf_task_labels == tt].long()
                    cur_task_labels = buf_labels[buf_task_labels == tt].long()
                    output_dict = self.model.forward(cur_task_inputs, cur_task_segment, cur_task_mask)
                    if 'dil' in self.args.scenario:
                        cur_task_output=output_dict['y']
                    elif 'til' in self.args.scenario:
                        outputs=output_dict['y']
                        cur_task_output = outputs[tt]
                    penalty = self.ce(cur_task_output, cur_task_labels)
                    penalty.backward()
                    self.store_grad(self.model.parameters, self.grads_cs[tt], self.grad_dims)

            # now compute the grad on the current data
            self.optimizer.zero_grad()
            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]

            loss = self.ce(output, targets)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

            # check if gradient violates buffer constraints
            if not self.buffer.is_empty():
                # copy gradient
                self.store_grad(self.model.parameters, self.grads_da, self.grad_dims)

                dot_prod = torch.mm(self.grads_da.unsqueeze(0),
                                torch.stack(self.grads_cs).T)
                if (dot_prod < 0).sum() != 0:
                    self.project2cone2(self.grads_da.unsqueeze(1),
                                  torch.stack(self.grads_cs).T, margin=self.args.gamma)
                    # copy gradients back
                    self.overwrite_grad(self.model.parameters, self.grads_da,
                                   self.grad_dims)

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




    def store_grad(self,params, grads, grad_dims):
        """
            This stores parameter gradients of past tasks.
            pp: parameters
            grads: gradients
            grad_dims: list with number of parameters per layers
        """
        # store the gradients
        grads.fill_(0.0)
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = np.sum(grad_dims[:count + 1])
                grads[begin: end].copy_(param.grad.data.view(-1))
            count += 1


    def overwrite_grad(self,params, newgrad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = sum(grad_dims[:count + 1])
                this_grad = newgrad[begin: end].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            count += 1


    def project2cone2(self,gradient, memories, margin=0.5, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.
            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
        """
        memories_np = memories.cpu().t().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        n_rows = memories_np.shape[0]
        self_prod = np.dot(memories_np, memories_np.transpose())
        self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
        grad_prod = np.dot(memories_np, gradient_np) * -1
        G = np.eye(n_rows)
        h = np.zeros(n_rows) + margin
        v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.from_numpy(x).view(-1, 1))
