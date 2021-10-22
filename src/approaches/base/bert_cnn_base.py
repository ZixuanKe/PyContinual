import sys,time
import numpy as np
import torch
# from copy import deepcopy
from copy import deepcopy
import torch.nn.functional as F

import utils
from tqdm import tqdm, trange
sys.path.append("./approaches/")
from contrastive_loss import SupConLoss, CRDLoss
from buffer import Buffer as Buffer
from itertools import zip_longest

class Appr(object):
    def __init__(self,model,logger,taskcla, args=None):
    # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        self.model=model
        # self.initial_model=deepcopy(model)
        self.nepochs=args.nepochs
        self.lr=args.lr
        self.lr_min=args.lr_min
        self.lr_factor=args.lr_factor
        self.lr_patience=args.lr_patience
        self.clipgrad=args.clipgrad
        self.args = args
        self.train_batch_size=args.train_batch_size
        self.eval_batch_size=args.eval_batch_size
        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.sup_con = SupConLoss(temperature=args.temp,base_temperature=args.base_temp)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()


        if 'ewc' in args.approach:
            self.model=model
            self.model_old=None
            self.fisher=None
            self.lamb=args.lamb  # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000

        if 'srk' in args.approach:
            self.control_1_s = torch.zeros(args.bert_hidden_size).cuda()
            self.control_2_s = torch.zeros(args.bert_hidden_size).cuda()
            self.control_3_s = torch.zeros(args.bert_hidden_size).cuda()

        if 'kan' in args.approach:
            self.smax = 400
            self.thres_cosh=50
            self.thres_emb=6
            self.lamb=0.75

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


        if 'owm' in args.approach:
            dtype = torch.cuda.FloatTensor  # run on GPU
            self.Pc1 = torch.autograd.Variable(torch.eye(100).type(dtype), volatile=True)
            self.Pc2 = torch.autograd.Variable(torch.eye(100).type(dtype), volatile=True)
            self.Pc3 = torch.autograd.Variable(torch.eye(100).type(dtype), volatile=True)
            self.P1 = torch.autograd.Variable(torch.eye(300).type(dtype), volatile=True)
            self.P2 = torch.autograd.Variable(torch.eye(300).type(dtype), volatile=True)
            self.test_max = 0

        if 'one' in args.approach:
            self.model=model
            self.initial_model=deepcopy(model)

        if 'hat' in args.approach:
            self.smax = 400  # Grid search = [140,200,300,400]; best was 400
            self.thres_cosh=50
            self.thres_emb=6
            self.lamb=0.75
            self.mask_pre=None
            self.mask_back=None

        if 'der' in args.approach:
            self.buffer = Buffer(self.args.buffer_size, self.device)
            self.mse = torch.nn.MSELoss()

        if 'gem' in args.approach:
            self.buffer = Buffer(self.args.buffer_size, self.device)
            # Allocate temporary synaptic memory
            self.grad_dims = []
            for pp in model.parameters():
                self.grad_dims.append(pp.data.numel())

            self.grads_cs = []
            self.grads_da = torch.zeros(np.sum(self.grad_dims)).to(self.device)
            self.logger = logger

        if 'a-gem' in args.approach:
            self.buffer = Buffer(self.args.buffer_size, self.device)
            self.grad_dims = []
            for param in self.model.parameters():
                self.grad_dims.append(param.data.numel())
            self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
            self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

        if 'l2' in args.approach:
            self.lamb=self.args.lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
            self.regularization_terms = {}
            self.task_count = 0
            self.online_reg = False  # True: There will be only one importance matrix and previous model parameters
                                    # False: Each task has its own importance matrix and model parameters


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

        print('CONTEXTUAL + KIM NCL')

        return


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



    def project(self,gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
        corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
        return gxy - corr * ger


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



    def sup_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets,t):
        if self.args.sup_head:
            outputs = torch.cat([output.clone().unsqueeze(1), output.clone().unsqueeze(1)], dim=1)
        else:
            outputs = torch.cat([pooled_rep.clone().unsqueeze(1), pooled_rep.clone().unsqueeze(1)], dim=1)

        loss = self.sup_con(outputs, targets,args=self.args)
        return loss


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


    def _get_optimizer_kan(self,lr=None,which_type=None):

        if which_type=='mcl':
            if lr is None: lr=self.lr
            if self.args.optimizer == 'sgd':
                return torch.optim.SGD(
                    [p for p in self.model.mcl.parameters()]+[p for p in self.model.last.parameters()],lr=lr)
            elif self.args.optimizer == 'adam':
                return torch.optim.Adam(
                    [p for p in self.model.mcl.parameters()]+[p for p in self.model.last.parameters()],lr=lr)

        elif which_type=='ac':
            if lr is None: lr=self.lr
            if self.args.optimizer == 'sgd':
                return torch.optim.SGD(
                    [p for p in self.model.ac.parameters()]+[p for p in self.model.last.parameters()],lr=lr)
            elif self.args.optimizer == 'adam':
                    return torch.optim.Adam(
                        [p for p in self.model.ac.parameters()]+[p for p in self.model.last.parameters()],lr=lr)



    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        if self.args.optimizer == 'sgd' and self.args.momentum:
            print('sgd+momentum')
            return torch.optim.SGD(self.model.parameters(),lr=lr, momentum=0.9,nesterov=True)
        elif self.args.optimizer == 'sgd':
            print('sgd')
            return torch.optim.SGD(self.model.parameters(),lr=lr)
        elif self.args.optimizer == 'adam':
            print('adam')
            return torch.optim.Adam(self.model.parameters(),lr=lr)


    def ent_id_detection(self,trained_task,input_ids, segment_ids, input_mask,t,which_type=None):

        output_d = {}

        outputs = []
        entropies = []

        if trained_task is None: #training
            entrop_to_test = range(0, t + 1)
        else: #testing
            entrop_to_test = range(0, trained_task + 1)

        for e in entrop_to_test:
            e_task=torch.LongTensor([e]).cuda()
            if 'hat' in self.args.approach:
                output_dict = self.model.forward(e_task,input_ids, segment_ids, input_mask,s=self.smax)
                masks = output_dict['masks']
                output_d['masks']= masks

            elif 'kan' in self.args.approach:
                output_dict = self.model.forward(e_task,input_ids, segment_ids, input_mask,which_type,s=self.smax)
            output = output_dict['y']
            outputs.append(output) #shared head

            Y_hat = F.softmax(output, -1)
            entropy = -1*torch.sum(Y_hat * torch.log(Y_hat))
            entropies.append(entropy)
        inf_task_id = torch.argmin(torch.stack(entropies))
        output=outputs[inf_task_id]

        output_d['output']= output

        return output_d




    def criterion_hat(self,outputs,targets,masks):
        reg=0
        count=0
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

    def criterion_ewc(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

        return self.ce(output,targets)+self.lamb*loss_reg



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