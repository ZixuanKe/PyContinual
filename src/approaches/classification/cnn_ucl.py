import sys, time, os
import numpy as np
import random
import torch
from copy import deepcopy
import utils
from utils import *
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import *
import math
from tqdm import tqdm, trange
sys.path.append('..')
from bayes_layer import BayesianLinear, BayesianConv2D, _calculate_fan_in_and_fan_out
sys.path.append("./approaches/base/")
from cnn_base import Appr as ApprBase


class Appr(ApprBase):
    

    def __init__(self,model,args=None,taskcla=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        print('CNN UCL NCL')


        return


    def train(self,t,train,valid,num_train_steps,train_data,valid_data):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        lr_rho = self.lr_rho
        patience = self.lr_patience
        self.optimizer = self._get_optimizer_ucl(lr, lr_rho)

        # Loop epochs
        for e in range(self.nepochs):
            self.epoch = self.epoch + 1
            # Train
            clock0 = time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')

            num_batch = len(train)
            
            self.train_epoch(t,train,iter_bar)
            
            clock1 = time.time()
            train_loss,train_acc,train_f1_macro=self.eval(t,train)

            clock2 = time.time()
            # print('time: ',float((clock1-clock0)*30*25))
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.train_batch_size * (clock1 - clock0) / num_batch,
                1000 * self.train_batch_size * (clock2 - clock1) / num_batch, train_loss, 100 * train_acc), end='')
            # Valid
            
            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

            # save log for current task & old tasks at every epoch
            # self.logger.add(epoch=(t * self.nepochs) + e, task_num=t + 1, valid_loss=valid_loss, valid_acc=valid_acc)


            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    lr_rho /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break

                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer_ucl(lr, lr_rho)
            print()

            utils.freeze_model(self.model_old)  # Freeze the weights
            
            
        # Restore best
        utils.set_model_(self.model, best_model)
        self.model_old = deepcopy(self.model)
        self.saved = 1

        # self.logger.save()

        return

    def train_epoch(self,t,data,iter_bar):
        self.model.train()
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            images,targets= batch

            # Forward current model
            mini_batch_size = len(targets)
            output_dict=self.model.forward(images)
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]


            loss=self.ce(output,targets)
            loss = self.custom_regularization(self.model_old, self.model, mini_batch_size, loss)
            # Backward
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            if self.args.optimizer == 'SGD' or self.args.optimizer == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,data,test=None,trained_task=None):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        target_list = []
        pred_list = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                images,targets= batch
                real_b=images.size(0)

                output_dict = self.model.forward(images, sample=False)
                # Forward
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

                _, pred = output.max(1)
                hits = (pred == targets).float()

                target_list.append(targets)
                pred_list.append(pred)


                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b
            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')


        return total_loss / total_num, total_acc / total_num,f1


# custom regularization

    def custom_regularization(self, saver_net, trainer_net, mini_batch_size, loss=None):
        
        sigma_weight_reg_sum = 0
        sigma_bias_reg_sum = 0
        sigma_weight_normal_reg_sum = 0
        sigma_bias_normal_reg_sum = 0
        mu_weight_reg_sum = 0
        mu_bias_reg_sum = 0
        L1_mu_weight_reg_sum = 0
        L1_mu_bias_reg_sum = 0
        
        out_features_max = 512
        alpha = self.args.alpha
        if self.saved:
            alpha = 1

        prev_weight_strength = nn.Parameter(torch.Tensor(3,1,1,1).uniform_(0,0))


        if 'mnist' in self.args.task:
            prev_weight_strength = nn.Parameter(torch.Tensor(self.args.image_size*self.args.image_size,1).uniform_(0,0))
        else:
            prev_weight_strength = nn.Parameter(torch.Tensor(3,1,1,1).uniform_(0,0))


        for (_, saver_layer), (_, trainer_layer) in zip(saver_net.named_children(), trainer_net.named_children()):
            if isinstance(trainer_layer, BayesianLinear)==False and isinstance(trainer_layer, BayesianConv2D)==False:
                continue
            # calculate mu regularization
            trainer_weight_mu = trainer_layer.weight_mu
            saver_weight_mu = saver_layer.weight_mu
            trainer_bias = trainer_layer.bias
            saver_bias = saver_layer.bias
            fan_in, fan_out = _calculate_fan_in_and_fan_out(trainer_weight_mu)
            
            trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
            saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))
            
            if isinstance(trainer_layer, BayesianLinear):
                std_init = math.sqrt((2 / fan_in) * self.args.ratio)
            if isinstance(trainer_layer, BayesianConv2D):
                std_init = math.sqrt((2 / fan_out) * self.args.ratio)
            
            saver_weight_strength = (std_init / saver_weight_sigma)

            if len(saver_weight_mu.shape) == 4:
                out_features, in_features, _, _ = saver_weight_mu.shape
                curr_strength = saver_weight_strength.expand(out_features,in_features,1,1)
                prev_strength = prev_weight_strength.permute(1,0,2,3).expand(out_features,in_features,1,1)
            
            else:
                out_features, in_features = saver_weight_mu.shape
                curr_strength = saver_weight_strength.expand(out_features,in_features)

                if len(prev_weight_strength.shape) == 4:
                    feature_size = in_features // (prev_weight_strength.shape[0])
                    prev_weight_strength = prev_weight_strength.reshape(prev_weight_strength.shape[0],-1)
                    prev_weight_strength = prev_weight_strength.expand(prev_weight_strength.shape[0], feature_size)
                    prev_weight_strength = prev_weight_strength.reshape(-1,1)

                prev_strength = prev_weight_strength.permute(1,0).expand(out_features,in_features)
            
            L2_strength = torch.max(curr_strength, prev_strength)
            bias_strength = torch.squeeze(saver_weight_strength)
            
            L1_sigma = saver_weight_sigma
            bias_sigma = torch.squeeze(saver_weight_sigma)
            
            prev_weight_strength = saver_weight_strength
            
            mu_weight_reg = (L2_strength * (trainer_weight_mu-saver_weight_mu)).norm(2)**2
            mu_bias_reg = (bias_strength * (trainer_bias-saver_bias)).norm(2)**2
            
            L1_mu_weight_reg = (torch.div(saver_weight_mu**2,L1_sigma**2)*(trainer_weight_mu - saver_weight_mu)).norm(1)
            L1_mu_bias_reg = (torch.div(saver_bias**2,bias_sigma**2)*(trainer_bias - saver_bias)).norm(1)
            
            L1_mu_weight_reg = L1_mu_weight_reg * (std_init ** 2)
            L1_mu_bias_reg = L1_mu_bias_reg * (std_init ** 2)
            
            weight_sigma = (trainer_weight_sigma**2 / saver_weight_sigma**2)
            
            normal_weight_sigma = trainer_weight_sigma**2
            
            sigma_weight_reg_sum = sigma_weight_reg_sum + (weight_sigma - torch.log(weight_sigma)).sum()
            sigma_weight_normal_reg_sum = sigma_weight_normal_reg_sum + (normal_weight_sigma - torch.log(normal_weight_sigma)).sum()
            
            mu_weight_reg_sum = mu_weight_reg_sum + mu_weight_reg
            mu_bias_reg_sum = mu_bias_reg_sum + mu_bias_reg
            L1_mu_weight_reg_sum = L1_mu_weight_reg_sum + L1_mu_weight_reg
            L1_mu_bias_reg_sum = L1_mu_bias_reg_sum + L1_mu_bias_reg
            
        # elbo loss
        loss = loss / mini_batch_size
        # L2 loss
        loss = loss + alpha * (mu_weight_reg_sum + mu_bias_reg_sum) / (2 * mini_batch_size)
        # L1 loss
        loss = loss + self.saved * (L1_mu_weight_reg_sum + L1_mu_bias_reg_sum) / (mini_batch_size)
        # sigma regularization
        loss = loss + self.beta * (sigma_weight_reg_sum + sigma_weight_normal_reg_sum) / (2 * mini_batch_size)
            
        return loss

