import time
import numpy as np
import torch
from tqdm import tqdm, trange
import utils
import torch.nn as nn
from contrastive_loss import SupConLoss, CRDLoss, DistillKL
from memory import ContrastMemory
from copy import deepcopy
import functools
import torch.nn.functional as F
import random
from buffer import Buffer as Buffer
from itertools import zip_longest
from contrastive_loss import SupConLoss, CRDLoss, MyContrastive,LabelSmoothingCrossEntropy
from copy import deepcopy

########################################################################################################################
# adapt from https://github.com/joansj/hat/blob/master/src/approaches/hat.py



#TODO: CNN based contrastive learning
class Appr(object):

    def __init__(self,model,aux_model=None,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,args=None,taskcla=None,logger=None):
        self.model=model
        self.aux_model=aux_model
        self.aux_nepochs=args.aux_nepochs
        self.nepochs=args.nepochs
        self.train_batch_size=args.train_batch_size
        self.eval_batch_size=args.eval_batch_size
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad
        self.smax = 400
        self.thres_cosh = 50
        self.thres_emb = 6
        self.lamb = 0.75
        self.mask_pre = None
        self.mask_back = None

        self.ce=torch.nn.CrossEntropyLoss()
        # self.optimizer=self._get_optimizer()
        # self.aux_optimizer=self._get_optimizer_aux()

        self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.sup_con = SupConLoss(temperature=args.temp,base_temperature=args.base_temp)
        self.model_old=None
        self.my_con = MyContrastive(args=args) # let's use a new version of contrastive loss


        if 'ucl' in args.approach:
            self.model_old = deepcopy(self.model)
            self.lr_rho = args.lr_rho
            self.lr_min = args.lr / (args.lr_factor ** 5)
            self.iteration = 0
            self.epoch = 0
            self.saved = 0
            self.beta = args.beta
            self.drop = [20,40,60,75,90]
            self.param_name = []

            for (name, p) in self.model.named_parameters():
                self.param_name.append(name)

        if 'one' in args.approach:
            self.model=model
            self.initial_model=deepcopy(model)

        if 'ewc' in args.approach:
            self.model=model
            self.model_old=None
            self.fisher=None
            self.lamb=args.lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000


        if 'owm' in args.approach:
            dtype = torch.cuda.FloatTensor  # run on GPU
            self.test_max = 0

            if 'cnn' in args.approach:
                self.Pc1 = torch.autograd.Variable(torch.eye(3 * 2 * 2).type(dtype), volatile=True)
                self.Pc2 = torch.autograd.Variable(torch.eye(64 * 2 * 2).type(dtype), volatile=True)
                self.Pc3 = torch.autograd.Variable(torch.eye(128 * 2 * 2).type(dtype), volatile=True)
                self.P1 = torch.autograd.Variable(torch.eye(256 * 4 * 4).type(dtype), volatile=True)
                self.P2 = torch.autograd.Variable(torch.eye(1000).type(dtype), volatile=True)

            elif 'mlp' in args.approach:
                self.P1 = torch.autograd.Variable(torch.eye(args.image_channel*args.image_size*args.image_size).type(dtype), volatile=True)
                self.P2 = torch.autograd.Variable(torch.eye(args.mlp_adapter_size).type(dtype), volatile=True)


        if 'der' in args.approach:
            self.buffer = Buffer(self.args.buffer_size, self.device)
            self.mse = torch.nn.MSELoss()

        if 'acl' in args.approach:
            self.buffer = Buffer(self.args.buffer_size, self.device)

        if 'hal' in args.approach:
            self.buffer = Buffer(self.args.buffer_size, self.device, self.args.ntasks, mode='ring')
            self.spare_model = deepcopy(model)
            self.finetuning_epochs = 1
            self.anchor_optimization_steps = 100
            self.spare_opt = torch.optim.SGD(self.spare_model.parameters(), lr=self.args.lr)

        if 'cat' in args.approach:
            self.smax = 400
            self.thres_cosh=50
            self.thres_emb=6
            self.lamb=0.75
            self.mask_pre=None
            self.mask_back=None

            self.acc_transfer=np.zeros((self.args.ntasks,self.args.ntasks),dtype=np.float32)
            self.acc_reference=np.zeros((self.args.ntasks,self.args.ntasks),dtype=np.float32)
            self.lss_transfer=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
            self.similarity_transfer=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)

            self.transfer_initial_model = deepcopy(self.model.transfer)

            self.check_federated = CheckFederated()
            self.history_mask_pre = []
            self.similarities = []

        if 'mtl' in args.approach:
            self.initial_model = deepcopy(model)

        print('CNN BASE')

        return

    def _get_optimizer_merge(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD([p for p in self.model.parameters()]+[p for p in self.aux_model.parameters()],lr=lr)


    def _get_optimizer_cat(self,lr=None,phase=None):
        if lr is None: lr=self.lr

        elif phase=='mcl' and 'no_attention' in self.args.loss_type:
            return torch.optim.SGD(list(self.model.mcl.parameters()),lr=lr)

        elif phase=='mcl' and 'joint' in self.args.loss_type:
            return torch.optim.SGD(list(self.model.kt.parameters())+list(self.model.mcl.parameters()),lr=lr)

        elif  phase=='transfer':
            return torch.optim.SGD(list(self.model.transfer.parameters()),lr=lr)

        elif  phase=='reference':
            return torch.optim.SGD(list(self.model.transfer.parameters()),lr=lr)



    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def _get_optimizer_aux(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.aux_model.parameters(),lr=lr)

    def _get_optimizer_owm(self, lr=None):
        # if lr is None:
        #     lr = self.lr
        lr = self.lr
        lr_owm = self.lr
        fc1_params = list(map(id, self.model.fc1.parameters()))
        fc2_params = list(map(id, self.model.fc2.parameters()))
        base_params = filter(lambda p: id(p) not in fc1_params + fc2_params,
                             self.model.parameters())
        optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': self.model.fc1.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc2.parameters(), 'lr': lr_owm},
                                     ], lr=lr, momentum=0.9)

        return optimizer

    def _get_optimizer_ucl(self, lr=None, lr_rho = None):
        if lr is None: lr = self.lr
        if lr_rho is None: lr_rho = self.lr_rho
        if self.args.optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, lr_rho=lr_rho, param_name = self.param_name)
        if self.args.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(),lr=lr)

    def til_output_friendly(self,tasks,outputs):
        output = []
        for t_id,t in enumerate(tasks): #different task in the same batch
            output.append(outputs[t][t_id].unsqueeze(0))
        output = torch.cat(output,0)
        return output


    def sup_loss(self,output,pooled_rep,images,targets):
        if self.args.sup_head:
            outputs = torch.cat([output.clone().unsqueeze(1), output.clone().unsqueeze(1)], dim=1)
        else:
            outputs = torch.cat([pooled_rep.clone().unsqueeze(1), pooled_rep.clone().unsqueeze(1)], dim=1)

        sup_loss = self.sup_con(outputs, targets,args=self.args)
        return sup_loss


    def augment_distill_loss(self,output,pooled_rep,images,targets, t,use_aux=False):
        augment_distill_loss = 0
        for pre_t in range(t):
            if self.args.distill_head:
                outputs = [output.clone().unsqueeze(1)]
            else:
                outputs = [pooled_rep.clone().unsqueeze(1)]

            with torch.no_grad(): #previous models are fixed in any time
                if use_aux:
                    pre_output_dict = self.aux_model(pre_t,images,s=self.smax)
                else:
                    pre_output_dict = self.model(pre_t,images,s=self.smax)
            pre_pooled_rep = pre_output_dict['normalized_pooled_rep']

            if self.args.distill_head_norm:
                pre_output = pre_output_dict['y']
                pre_output_rep = F.normalize(pre_output, dim=1)
            else:
                pre_output = pre_output_dict['y']

            if self.args.distill_head: #append everyone
                if self.args.distill_head_norm:
                    outputs.append(pre_output_rep.unsqueeze(1).clone())
                else:
                    outputs.append(pre_output.unsqueeze(1).clone())
            else:
                outputs.append(pre_pooled_rep.unsqueeze(1).clone())

            outputs = torch.cat(outputs, dim=1)
            augment_distill_loss+= self.sup_con(outputs,args=self.args) #sum up all distillation

        return augment_distill_loss



    def amix_loss(self,output,pooled_rep,images,targets, t,s,use_aux=False,fix_aux=False):

        #s1: train hat
        #s2: train aux: aux
        #s3: train aux: base

        amix_loss = 0
        if self.args.amix_head:
            mix_pooled_reps = [output.clone().unsqueeze(1)]
        else:
            mix_pooled_reps = [pooled_rep.clone().unsqueeze(1)]

        if self.args.attn_type == 'self':
            orders = [[pre_t for prre_t in range(t)]]

            for order_id,order in enumerate(orders):
                if use_aux and fix_aux:
                    with torch.no_grad():
                        print('use and fix aux')
                        mix_output_dict = self.aux_model(t,images,s=s,start_mixup=True,l=order,idx=order_id,mix_type=self.args.mix_type)

                elif use_aux:
                    # print('train aux')
                    mix_output_dict = self.aux_model(t,images,s=s,start_mixup=True,l=order,idx=order_id,mix_type=self.args.mix_type)

                else:
                    mix_output_dict = self.model(t,images,s=s,start_mixup=True,l=order,idx=order_id,mix_type=self.args.mix_type)
                mix_output = mix_output_dict['y']
                mix_masks = mix_output_dict['masks']
                mix_pooled_rep = mix_output_dict['normalized_pooled_rep']

                if 'til' in self.args.scenario:
                    mix_output = mix_output[t]

                n_loss,_=self.criterion_hat(mix_output,targets,mix_masks) # it self is also training
                amix_loss+=n_loss # let's first do some pre-training


                if self.args.amix_head:
                    if self.args.amix_head_norm:
                        mix_output_rep = F.normalize(mix_output, dim=1)
                        mix_pooled_reps.append(mix_output_rep.unsqueeze(1).clone())
                    else:
                        mix_pooled_reps.append(mix_output.unsqueeze(1).clone())

                else:
                    mix_pooled_reps.append(mix_pooled_rep.unsqueeze(1).clone())

        cur_mix_outputs = torch.cat(mix_pooled_reps, dim=1)

        amix_loss += self.sup_con(cur_mix_outputs, targets,args=self.args) #train attention and contrastive learning at the same time
        return amix_loss



    def ent_id_detection(self,trained_task,images,t):

        output_d = {}

        outputs = []
        outputs_attn= []
        entropies = []

        if trained_task is None: #training
            entrop_to_test = range(0, t + 1)
        else: #testing
            entrop_to_test = range(0, trained_task + 1)
        # print('entrop_to_test: ',list(entrop_to_test))
        for e in entrop_to_test:
            if 'acl' in self.args.approach:
                task = torch.LongTensor([e]).repeat(images.size(0))
                tt=task.to(device=self.device)
                output_dict=self.model(images, images, tt, trained_task)

                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    output = output_dict['y'][e]

            elif 'hat_merge' in self.args.approach:
                output_dict = self.model.forward(images)

                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    output = output_dict['y'][e]

            elif 'hat' in self.args.approach:
                output_dict = self.model.forward(e,images,s=self.smax)
                masks = output_dict['masks']
                output_d['masks']= masks

                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    output = output_dict['y'][e]

            elif 'ucl' in self.args.approach:
                output_dict = self.model.forward(images,sample=False)
                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    output = output_dict['y'][e]

            elif 'cat' in self.args.approach:
                output_dict = self.model.forward(e,images,s=self.smax,phase='mcl',similarity=self.similarities[-1],
                                                        history_mask_pre=self.history_mask_pre,check_federated=self.check_federated)
                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                    output_attn = output_dict['y_attn']

                elif 'til' in self.args.scenario:
                    output = output_dict['y'][e]
                    output_attn = output_dict['y_attn'][e]

                outputs_attn.append(output_attn) #shared head

                masks = output_dict['masks']
                output_d['masks']= masks
            else:
                #In TIL setting, we want to know which head to use
                output_dict = self.model.forward(images)
                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    output =  output_dict['y'][e]

            outputs.append(output) #shared head

            Y_hat = F.softmax(output, -1)
            entropy = -1*torch.sum(Y_hat * torch.log(Y_hat))
            entropies.append(entropy)
        inf_task_id = torch.argmin(torch.stack(entropies))
        # print('inf_task_id: ',inf_task_id)
        output=outputs[inf_task_id]
        if 'cat' in self.args.approach:
            output_attn = outputs_attn[inf_task_id]
            output_d['output_attn']= output_attn

        output_d['output']= output

        return output_d


    def criterion_ewc(self,t,output,targets):
        # Regularization for all previous tasks
        if self.args.eval_only:
            return self.ce(output,targets)
        else:
            loss_reg=0
            if t>0:
                for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                    loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

            return self.ce(output,targets)+self.lamb*loss_reg

    def criterion_hat(self,outputs,targets,masks,t=None):
        reg=0
        count=0
        ewc_loss=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()

        print('reg: ',reg)
        print('count: ',count)
        reg/=count

        return self.ce(outputs,targets)+self.lamb*reg,reg

    def f1_compute_fn(self,y_true, y_pred,average):
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return f1_score(y_true, y_pred,average=average)
########################################################################################################################
class CheckFederated():
    def __init__(self):
        pass
    def set_similarities(self,similarities):
        self.similarities = similarities

    def fix_length(self):
        return len(self.similarities)

    def get_similarities(self):
        return self.similarities


    def check_t(self,t):
        if t < len([sum(x) for x in zip_longest(*self.similarities, fillvalue=0)]) and [sum(x) for x in zip_longest(*self.similarities, fillvalue=0)][t] > 0:
            return True

        elif np.count_nonzero(self.similarities[t]) > 0:
            return True

        elif t < len(self.similarities[-1]) and self.similarities[-1][t] == 1:
            return True

        return False