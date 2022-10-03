

import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import torchvision
import numpy as np
import math
from typing import Optional, Tuple

from tqdm.notebook import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


# Utility functions
def set_model_specific_task(model, specific_task, verbose=False):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            if verbose:
                print(f"=> Set specific_task of {n} to {specific_task}")

            m.specific_task = specific_task

def set_model_share_task(model, share_task):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            m.share_task = share_task

def set_model_generative_task(model, gen_task):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            m.gen_task = gen_task

def set_model_pool_task(model, pool_task):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            m.pool_task = pool_task



def set_alphas(model, alphas):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            m.alphas = alphas



def set_model_sim(model, sim):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            m.sim = sim


def set_model_oneshot(model, oneshot):
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            m.oneshot = oneshot

def model_weight_copy(model, source_id, target_id):
    print('copy from ' + str(source_id) + ' to ' + str(target_id))
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            m.specific_scores[target_id] = m.specific_scores[source_id]



def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
    gain = nn.init.calculate_gain('relu')
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std


def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores


def sim_matrix(a, b, eps=1e-8):
    """Batch version of CosineSimilarity."""
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)

    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


# Subnetwork forward from hidden networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        return (scores >= 0).float() # use 0 as threshold. this is related to the signed_constant initialization

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass. so that it is trainable
        return g



class MultitaskMaskLinear(nn.Linear):

    def _select_mask_ggg(self):
        if self.sim == 'generative':  # where we know the task

            if torch.is_tensor(self.gen_task):
                cat_gen_scores = []
                for t in self.gen_task:
                    cat_gen_scores.append(self.gen_scores[t])
                selected_mask = torch.stack(cat_gen_scores)

            else:
                selected_mask = self.gen_scores[self.gen_task]

        elif self.sim == 'pool' and 'reconstruct' in self.args.baseline:
            if torch.is_tensor(self.pool_task):
                cat_pool_scores = []
                for t in self.pool_task:
                    cat_pool_scores.append(self.pool_scores[t])
                selected_mask = torch.stack(cat_pool_scores)

            else:
                selected_mask = self.pool_scores[self.pool_task]

        elif self.sim == 'pool':
            if torch.is_tensor(self.specific_task):
                cat_specific_scores = []
                for t in self.pool_task:
                    cat_specific_scores.append(self.specific_scores[t])
                selected_mask = torch.stack(cat_specific_scores)

            else:
                selected_mask = self.specific_scores[self.pool_task]



        elif hasattr(self, 'share_scores'):
            # print('share: ',self.share_task)

            if torch.is_tensor(self.share_task):
                cat_share_scores = []
                for t in self.share_task:
                    cat_share_scores.append(self.share_scores[t])
                selected_mask = torch.stack(cat_share_scores)

                # cat_share_scores = torch.stack([_ for _ in self.share_scores])
                # selected_mask = cat_share_scores.index_select(0, self.share_task) # different instance may choose differently
            else:
                selected_mask = self.share_scores[self.share_task]
        elif hasattr(self, 'specific_scores'):
            # print('specific: ',self.specific_task)
            if torch.is_tensor(self.specific_task):
                cat_specific_scores = []
                for t in self.specific_task:
                    cat_specific_scores.append(self.specific_scores[t])
                selected_mask = torch.stack(cat_specific_scores)
                # cat_specific_scores = torch.stack([_ for _ in self.specific_scores])
                # selected_mask = cat_specific_scores.index_select(0, self.specific_task)
                # different instance may choose differently
                # TODO: which I am not familar with
                # selected_mask = selected_mask.squeeze()
            else:
                selected_mask = self.specific_scores[self.specific_task]
            # print('selected_mask: ',selected_mask.size())
        else:
            raise NotImplementedError

        return selected_mask



    def __init__(self, *args, num_tasks=1,adapter_role=None, N=1, my_args=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.pool_size = my_args.pool_size
        self.M = my_args.ntasks
        self.N = N
        # self.lam = 0.5  # Follow the original paper.
        self.matching_loss = None
        self.args = my_args
        self.hidden_size = 1024

        if ('mtl' in my_args.baseline or 'backward' in my_args.baseline) and adapter_role=='up': # up for specific
            self.specific_scores = nn.ParameterList(
                [
                    nn.Parameter(mask_init(self)) #TODO: consider a shared mask pool. Make it clear how to achieve
                    for _ in range(num_tasks)
                ]
            )

        elif ('mtl' in my_args.baseline or 'backward' in my_args.baseline) and adapter_role == 'down': # down for share
            self.share_scores = nn.ParameterList(
                [
                    nn.Parameter(mask_init(self)) #TODO: consider a shared mask pool. Make it clear how to achieve
                    for _ in range(num_tasks)
                ]
            )

        else:
            self.specific_scores = nn.ParameterList(
                [
                    nn.Parameter(mask_init(self)) #TODO: consider a shared mask pool. Make it clear how to achieve
                    for _ in range(num_tasks)
                ]
            )

        if 'generative' in self.args.baseline:
            self.gen_scores = nn.ParameterList(
                [
                    nn.Parameter(mask_init(self)) #TODO: consider a shared mask pool. Make it clear how to achieve
                    for _ in range(num_tasks)
                ]
            )

        if 'pool' in self.args.baseline and 'reconstruct' in self.args.baseline:
            self.pool_scores = nn.ParameterList(
                [
                    nn.Parameter(mask_init(self)) #TODO: consider a shared mask pool. Make it clear how to achieve
                    for _ in range(num_tasks)
                ]
            )



        signed_constant(self)


    def forward(self,x): # no need to use residual_input
        # return F.linear(x, self.weight, self.bias)
        self.num_tasks_learned = self.args.ntasks
        if hasattr(self, 'oneshot') and self.oneshot:

            if 'reconstruct' in self.args.baseline:

                self.stacked = torch.stack(
                    [
                        GetSubnet.apply(self.pool_scores[j])
                        for j in range(self.num_tasks)
                    ]
            )

            else:
                self.stacked = torch.stack(
                    [
                        GetSubnet.apply(self.specific_scores[j])
                        for j in range(self.num_tasks)
                    ]
                )


            # Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs]
                * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)

            w = self.weight * subnet
            x = F.linear(x, w, self.bias)

        else:
            selected_mask = self._select_mask_ggg()
            subnet = GetSubnet.apply(selected_mask)
            w = self.weight * subnet
            # print('w: ',w.size())

            if len(w.size()) > 2:  # differnt instance may have different weight
                result_x = []
                for ins in range(len(x)):
                    result_x.append(F.linear(x, w[ins], self.bias)[ins])
                x = torch.stack(result_x)
            else:
                x = F.linear(x, w, self.bias)

        return x


    def __repr__(self):
        return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"

