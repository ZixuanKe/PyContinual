import logging
import math

import numpy as np
import os
import torch
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from utils import utils
from networks.buffer import Buffer
from networks.baselines import ewc, hat, cat


        # before training ***********************************************************************************************
def prepare(self, model, train_loader, dev_loader, accelerator):

    #TODO: consider separate the baselines
    mask_pre = None
    mask_back = None
    self_fisher = None
    self.args.is_cat = False

    #need to adapt everything to MTL. There is no separate head, so no changes are neeeded here
    if 'ewc' in self.args.baseline:

        if os.path.exists(os.path.join(self.args.prev_output, 'fisher')):
            print('load fisher matrix from ' + self.args.prev_output + ' **************')
            self_fisher = torch.load(os.path.join(self.args.prev_output, 'fisher'), map_location=torch.device('cpu'))
            for k,v in self_fisher.items():
                self_fisher[k] = self_fisher[k].cuda()

    # replay baselines TODO: use only one 'if'
    elif 'ldbr' in self.args.baseline or 'derpp' in self.args.baseline or 'agem' in self.args.baseline:
        if self.args.ft_task == 0:
            buffer = Buffer(self.args.buffer_size_per_dataset * self.args.ntasks, accelerator.device,)
            if 'agem' in self.args.baseline:
                self.args.grad_dims = []
                for param in model.model.parameters():
                    self.args.grad_dims.append(param.data.numel())
                self.args.grad_xy = torch.Tensor(np.sum(self.args.grad_dims)).to(accelerator.device)
                self.args.grad_er = torch.Tensor(np.sum(self.args.grad_dims)).to(accelerator.device)
        else:
            buffer = torch.load(os.path.join(self.args.prev_output, 'buffer'), map_location=accelerator.device)
            if 'agem' in self.args.baseline:
                self.args.grad_dims = torch.load(os.path.join(self.args.prev_output, 'grad_dims'), map_location=accelerator.device)
                self.args.grad_xy = torch.load(os.path.join(self.args.prev_output, 'grad_xy'), map_location=accelerator.device)
                self.args.grad_er = torch.load(os.path.join(self.args.prev_output, 'grad_er'), map_location=accelerator.device)
        self.args.buffer = buffer

    elif 'adapter_hat' in self.args.baseline   \
            or 'adapter_cat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_ctr' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:  # BCL included HAT

        self.args.eval_t = self.args.ft_task

        if os.path.exists(os.path.join(self.args.prev_output, 'mask_pre')):
            print('load mask matrix **************')

            mask_pre = torch.load(os.path.join(self.args.prev_output, 'mask_pre'))
            mask_back = torch.load(os.path.join(self.args.prev_output, 'mask_back'))

            for k, v in mask_pre.items():
                mask_pre[k] = mask_pre[k].cuda()

            for k, v in mask_back.items():
                mask_back[k] = mask_back[k].cuda()

        if 'adapter_cat' in self.args.baseline: # initialize the new adapter using the nearest adapter

            if self.args.eval_only:
                if os.path.exists(os.path.join(self.args.output_dir, 'similarities')):
                    similarities = torch.load(os.path.join(self.args.output_dir, 'similarities'),
                                              map_location=torch.device('cpu'))
                    self.similarity.similarities = similarities

                self.args.similarity = self.similarity

            else:

                if self.args.known_similarity: # this is simplu for testing

                    self.similarity.set_similarities([0])
                    self.similarity.set_similarities([0])
                    self.similarity.set_similarities([0,0])
                    self.similarity.set_similarities([0,0,0])
                    self.similarity.set_similarities([0,0,0,0])
                    self.args.similarity = self.similarity

                else:
                    if os.path.exists(os.path.join(self.args.prev_output, 'similarities')):
                        similarities = torch.load(os.path.join(self.args.prev_output, 'similarities'),
                                                  map_location=torch.device('cpu'))
                        self.similarity.similarities = similarities


                    if self.args.ft_task == 0:
                        self.similarity.set_similarities([0])
                        self.args.similarity = self.similarity

                    else:
                        similarity = cat.compute(self, model, train_loader, dev_loader, accelerator)

                        self.similarity.set_similarities(similarity)
                        self.args.similarity = self.similarity

            print('similarity: ',self.args.similarity.similarities)

    elif 'l2p' in self.args.baseline:
        self.args.n_tokens = self.args.N * self.args.Lp
    
    metric = utils.load_my_metric(self.args)

    return self, model, train_loader, dev_loader, accelerator,metric,mask_pre,mask_back,self_fisher
