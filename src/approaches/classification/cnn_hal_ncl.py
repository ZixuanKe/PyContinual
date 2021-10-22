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


#HAL: adapt from https://github.com/aimagelab/mammoth/blob/master/models/hal.py

#TODO: where is bi-level optimization?

class Appr(ApprBase):

    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        print('CNN HAL NCL')


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

        self.task_number = t
        # ring buffer mgmt (if we are not loading
        if self.task_number > self.buffer.task_number:
            self.buffer.num_seen_examples = 0
            self.buffer.task_number = self.task_number
        # get anchors (provided that we are not loading the model
        if len(self.anchors) < self.task_number * self.args.nclasses:
            self.get_anchors(t,train)
            del self.phi


        samples_per_task = int(len(train_data) * self.args.buffer_percent)
        loader = DataLoader(train_data, batch_size=samples_per_task)

        images,targets = next(iter(loader))
        images = images.to(self.device)
        targets = targets.to(self.device)


        self.buffer.add_data(
            examples=images,
            labels=targets,
            task_labels=torch.ones(samples_per_task,dtype=torch.long).to(self.device) * (t),
            segment_ids=images, #dumy
            input_mask=images,
        )


        return

    def get_anchors(self, t,dataset):
        theta_t = self.model.get_params().detach().clone()
        self.spare_model.set_params(theta_t)

        # fine tune on memory buffer
        for _ in range(self.finetuning_epochs):
            buf_inputs, buf_labels, buf_task_labels, buf_segment_ids,buf_input_mask = self.buffer.get_data(
                self.args.buffer_size)
            inputs = buf_inputs
            labels = buf_labels

            self.spare_opt.zero_grad()
            output_dict = self.spare_model(inputs)
            if 'dil' in self.args.scenario:
                out=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                out = outputs[t]


            loss = self.ce(out, labels)
            loss.backward()
            self.spare_opt.step()

        theta_m = self.spare_model.get_params().detach().clone()

        classes_for_this_task = range(self.args.nclasses) #for scenrios other than DIL, you need to check he website for adaptation

        for a_class in classes_for_this_task:
            e_t = torch.rand(self.input_shape, requires_grad=True, device=self.device)
            e_t_opt = torch.optim.SGD([e_t], lr=self.args.lr)
            print(file=sys.stderr)
            for i in range(self.anchor_optimization_steps):
                e_t_opt.zero_grad()
                cum_loss = 0

                self.spare_opt.zero_grad()
                self.spare_model.set_params(theta_m.detach().clone())
                output_dict = self.spare_model(e_t.unsqueeze(0))
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]

                loss = -torch.sum(self.ce(output, torch.tensor([a_class]).to(self.device)))
                loss.backward()
                cum_loss += loss.item()

                self.spare_opt.zero_grad()
                self.spare_model.set_params(theta_t.detach().clone())

                output_dict = self.spare_model(e_t.unsqueeze(0))
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]

                loss = torch.sum(self.ce(output, torch.tensor([a_class]).to(self.device)))
                loss.backward()
                cum_loss += loss.item()

                self.spare_opt.zero_grad()
                loss = torch.sum(self.args.gamma * (self.spare_model.features(e_t.unsqueeze(0)) - self.phi) ** 2)
                assert not self.phi.requires_grad
                loss.backward()
                cum_loss += loss.item()

                e_t_opt.step()

            e_t = e_t.detach()
            e_t.requires_grad = False
            self.anchors = torch.cat((self.anchors, e_t.unsqueeze(0)))
            del e_t
            print('Total anchors:', len(self.anchors), file=sys.stderr)

        self.spare_model.zero_grad()




    def train_epoch(self,t,data,iter_bar):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                t.to(self.device) if t is not None else None for t in batch]
            images,targets= batch


            real_batch_size = images.shape[0]
            if not hasattr(self, 'input_shape'):
                self.input_shape = images.shape[1:]
            if not hasattr(self, 'anchors'):
                self.anchors = torch.zeros(tuple([0] + list(self.input_shape))).to(self.device)
            if not hasattr(self, 'phi'):
                print('Building phi', file=sys.stderr)
                with torch.no_grad():
                    self.phi = torch.zeros_like(self.model.features(images[0].unsqueeze(0)), requires_grad=False)
                assert not self.phi.requires_grad

            if not self.buffer.is_empty(): # ACL does use memeory
                buf_inputs, buf_labels, buf_task_labels, buf_segment_ids,buf_input_mask = self.buffer.get_data(
                    self.args.buffer_size)
                images = torch.cat((images, buf_inputs))
                targets = torch.cat((targets, buf_labels))


            old_weights = self.model.get_params().detach().clone()

            self.optimizer.zero_grad()
            output_dict = self.model(images)
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]
            loss=self.ce(output,targets)
            loss.backward()
            self.optimizer.step()

            # print('self.anchors: ',self.anchors.size())
            # assert len(self.anchors) == self.args.nclasses #for scenrios other than DIL, you need to check he website for adaptation

            if len(self.anchors) > 0:
                with torch.no_grad():
                    output_dict = self.model(self.anchors)
                    if 'dil' in self.args.scenario:
                        output=output_dict['y']
                    elif 'til' in self.args.scenario:
                        outputs=output_dict['y']
                        output = outputs[t]

                    pred_anchors = output

                self.model.set_params(old_weights)
                output_dict = self.model(self.anchors)
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]

                pred_anchors -= output
                loss = self.args.hal_lambda * (pred_anchors ** 2).mean()
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.phi = self.args.beta * self.phi + (1 - self.args.beta) * self.model.features(images[:real_batch_size]).mean(0)

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
                output = output_dict['y']
                if 'dil' in self.args.scenario:
                    output=output_dict['y'] # notthing to do with id
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
