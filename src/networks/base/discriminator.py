# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import utils

class Discriminator(torch.nn.Module): #discriminator network
    def __init__(self,args,t):
        super(Discriminator, self).__init__()

        ncha = args.image_channel
        size = args.image_size

        if 'mlp' in args.approach:
            nhid=args.mlp_adapter_size #if nlp
        else:
            nhid=2048 #if nlp

        if args.diff == 'yes':
            self.dis = torch.nn.Sequential(
                GradientReversal(args.lam),
                torch.nn.Linear(nhid, nhid),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(nhid, nhid),
                torch.nn.Linear(nhid, t + 2)
            )
        else:
            self.dis = torch.nn.Sequential(
                torch.nn.Linear(nhid, nhid),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(nhid, nhid),
                torch.nn.Linear(nhid, t + 2)
            )


    def forward(self, z, labels, task_id):
        return self.dis(z)

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


    def get_size(self):
        count=sum(p.numel() for p in self.dis.parameters() if p.requires_grad)
        print('Num parameters in D       = %s ' % (self.pretty_print(count)))


class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26

    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)