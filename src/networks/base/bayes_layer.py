import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _single, _pair, _triple

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu.cuda()
        self.rho = rho.cuda()
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.mu.size()).cuda()
        return self.mu + self.sigma * epsilon   

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        gain = 1 # Var[w] + sigma^2 = 2/fan_in
        
        total_var = 2 / fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var
        
        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std)-1)
        
        nn.init.uniform_(self.weight_mu, -bound, bound)
        self.bias = nn.Parameter(torch.Tensor(out_features).uniform_(0,0))
        
        self.weight_rho = nn.Parameter(torch.Tensor(out_features,1).uniform_(rho_init,rho_init))
        
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, input, sample=False):
        if sample:
            weight = self.weight.sample()
            bias = self.bias
        else:
            weight = self.weight.mu
            bias = self.bias

        return F.linear(input, weight, bias)


class _BayesianConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride,padding, dilation, transposed, output_padding, groups, bias, ratio):
        super(_BayesianConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        # print(kernel_size)
        if hasattr(kernel_size, '__iter__'):
            self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        else:
            self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size))

        # print(self.weight_mu.size())

        _, fan_out = _calculate_fan_in_and_fan_out(self.weight_mu)
        total_var = 2 / fan_out
        noise_var = total_var * ratio
        mu_var = total_var - noise_var
        
        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std)-1)
        
        nn.init.uniform_(self.weight_mu, -bound, bound)
        self.bias = nn.Parameter(torch.Tensor(out_channels).uniform_(0,0), requires_grad = bias)
        
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1).uniform_(rho_init,rho_init))
        
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        
        
class BayesianConv2D(_BayesianConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, ratio=0.25):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesianConv2D, self).__init__(in_channels, out_channels, kernel_size, 
                                             stride, padding, dilation, False, _pair(0), groups, bias, ratio)
    
    def forward(self, input, sample = False):
        if sample:
            weight = self.weight.sample()
            
        else:
            weight = self.weight.mu
        
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



