import torch
from torch.optim import Optimizer
from modules.quant_function import QuantConv2d
import torch.nn as nn
from torch.optim import Adam,SGD
import numpy as np

def log_t(t, gamma, T_max, eps=1e-6):
    normalized_log = np.log(t+1) / (np.log(T_max+1) + eps)
    modulation = 1 - np.exp(-5 * normalized_log)
    return gamma * normalized_log * modulation

class Bop(Optimizer): 
    """self design Bop from Rethinking Binarized Neural Network Optimization arXiv:1906.02107,
    Need to implement method to flip weight to avoid updating latent weight.
    Args:
        params(iterable): iterable of parameters to optimize or dicts defining parameter group
        lr(float): learning rate for non-quantcConv
        threshold: determines to whether to flip each weight.
        gamma: the adaptivity rate.
    Example:
        >>> from Bop import *
        >>> optimizer = Bop(model.parameters(), lr=0.1,  threshold=1e-7, gamma=0.999)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, model, lr=1e-4, threshold=1e-9, beta1=1e-3,beta2=1e-5,weight_decay=0):
        quantconv_params = [] 
        other_params = [] 
        for m in model.modules():
            if isinstance(m, QuantConv2d):
                quantconv_params.append(m.weight)
            else:
                if hasattr(m, 'weight'):
                    other_params.append(m.weight)
                if hasattr(m,'bias'):
                    other_params.append(m.bias)
        self.t = 1
        self._adam = SGD(other_params, lr=lr)
        defaults = dict(lr=lr, threshold=threshold, beta1=beta1,beta2=beta2,weight_decay=weight_decay)
        super(Bop,self).__init__(quantconv_params,defaults=defaults)
    
    def __setstate__(self, state):
        super(Bop, self).__setstate__(state)

    def step(self,t,t_max, reset=False, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._adam.step()
        loss = None
        eps = 1e-10
        if closure is not None:
            loss = closure()
                    
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'm' not in state:
                    state['m'] = torch.zeros_like(p.data)
                if 'v' not in state:
                    state['v'] = torch.zeros(t_max)
                    
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                beta1 = group['beta1']
                beta2 = group['beta2']
                if reset:
                    state['m'] = (1 -beta1) * state['m'] + beta1 * d_p
                    reverse = -torch.sign(p.data * state['m'] - group['threshold'])
                    p.data.copy_(torch.sign(reverse * p.data))
                    state['m'].copy_((reverse + 1) / 2 * state['m'])
                else:
                    state['m'].copy_((1 - beta1) * state['m'] + beta1 * d_p)
                    state['v'][t] = (1 - beta2) * state['v'][t] + torch.mean(beta2 * torch.mul(d_p,d_p))
                    reverse = -torch.sign(p.data * state['m'] / (torch.sqrt(state['v'][t]) + eps) - group['threshold'])
                    p.data.copy_(torch.sign(reverse * p.data))

        self.t += 1
        return loss
    def zero_grad(self) -> None:
        super().zero_grad()
        self._adam.zero_grad()
        

class CustomScheduler():
    def __init__(self, optimizer:Optimizer,param_name: str, decay_epochs=50, decay=0.1):
        self.decay_epochs = decay_epochs
        self.decay = decay
        self.optimizer = optimizer
        self.param_name = param_name
        
    def step(self, epoch):
        if epoch % self.decay_epochs == 0 and epoch > 0:
            for group in self.optimizer.param_groups:
                if group[self.param_name]:
                    group[self.param_name] *= self.decay