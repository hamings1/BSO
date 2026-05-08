import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models import create_model

import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
from modules.quant_function import QuantConv2d

__all__ = ['birealnet19', 'birealnet18', 'birealnet34','QuantConv2d']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Replace(Function):
    @staticmethod
    def forward(ctx, z1, z1_r):
        return z1_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)


class WrapedSNNOp(nn.Module):

    def __init__(self, op):
        super(WrapedSNNOp, self).__init__()
        self.op = op

    def forward(self, x, **kwargs):
        require_wrap = kwargs.get('require_wrap', True)
        if require_wrap:
            B = x.shape[0] // 2
            spike = x[:B]
            rate = x[B:]
            with torch.no_grad():
                out = self.op(spike).detach()
            in_for_grad = Replace.apply(spike, rate)
            out_for_grad = self.op(in_for_grad)
            output = Replace.apply(out_for_grad, out)
            return output
        else:
            return self.op(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, weight_standardization=False, beta=1.0, alpha=1.0,
                 single_step_neuron: callable = None, stochdepth_rate=0.0, **kwargs):
        super(BasicBlock, self).__init__()
        self.stochdepth_rate = stochdepth_rate
        self.grad_with_rate = kwargs.get('grad_with_rate', False)

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        self.conv = conv3x3(inplanes, planes, stride)
        self.bn = nn.BatchNorm2d(planes)  
        self.sn = single_step_neuron(**kwargs)

        self.downsample = downsample
        self.stride = stride

        self.beta, self.alpha = beta, alpha
        self.skipinit_gain = nn.Parameter(torch.zeros(()))
        if self.grad_with_rate:
            self.conv = WrapedSNNOp(self.conv)
            if self.downsample != None:
                self.downsample = WrapedSNNOp(self.downsample)



    def forward(self, x, **kwargs):
        require_wrap = self.grad_with_rate and self.training
        if require_wrap:
            out = self.sn(x, output_type='spike_rate', **kwargs)
        else:
            out = self.sn(x, **kwargs)

        if self.downsample is not None:
            if require_wrap:
                identity = self.downsample(x, require_wrap=False)
            else:
                if self.grad_with_rate:
                    identity = self.downsample(x, require_wrap=False)
                else:
                    identity = self.downsample(x)
        else:
            identity = x

        if require_wrap:
            out = self.conv(out, require_wrap=True)
            out = self.bn(out)
        else:
            if self.grad_with_rate:
                out = self.conv(out, require_wrap=False)
            else:
                out = self.conv(out)
            out = self.bn(out)

        out += identity

        return out

    def get_spike(self):
        spikes = []
        spike = self.sn.spike.cpu()
        spikes.append(spike.reshape(spike.shape[0], -1))
        return spikes

class SequentialModule(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self._modules.values():
            input = module(input, **kwargs)
        return input

    def get_spike(self):
        spikes = []
        for module in self._modules.values():
            spikes_module = module.get_spike()
            spikes += spikes_module
        return spikes

class BiRealNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, 
                groups=1, width_per_group=64, replace_stride_with_dilation=None,
                weight_standardization=False, single_step_neuron: callable = None,
                alpha=0.2, drop_rate=0.0,pretrained_cfg=None, **kwargs):
        super(BiRealNet, self).__init__()
        self.ws = weight_standardization
        self.inplanes = 64
        self.dilation = 1
        self.alpha = alpha
        self.groups = groups
        self.base_width = width_per_group
        self.c_in = kwargs.get('c_in', 3)
        replace_stride_with_dilation = [False, False, False]
        self.conv = nn.Conv2d(self.c_in, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn = nn.BatchNorm2d(self.inplanes)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        expected_var = 1.0
        self.layer1  = self._make_layer(block, 64, layers[0], alpha=self.alpha, var=expected_var, single_step_neuron=single_step_neuron, **kwargs)
        self.layer2  = self._make_layer(block, 128, layers[1], stride=2, alpha=self.alpha, var=expected_var,
                                                     dilate=replace_stride_with_dilation[0], single_step_neuron=single_step_neuron, **kwargs)
        self.layer3  = self._make_layer(block, 256, layers[2], stride=2, alpha=self.alpha, var=expected_var,
                                                     dilate=replace_stride_with_dilation[1], single_step_neuron=single_step_neuron, **kwargs)
        self.layer4  = self._make_layer(block, 512, layers[3], stride=2, alpha=self.alpha, var=expected_var,
                                                     dilate=replace_stride_with_dilation[2], single_step_neuron=single_step_neuron, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1,alpha=1.0, var=1.0, dilate=False, single_step_neuron: callable = None, **kwargs):
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = SequentialModule(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        beta = var ** 0.5
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, self.ws, beta, alpha, single_step_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                weight_standardization=self.ws, beta=beta, alpha=alpha, single_step_neuron=single_step_neuron, **kwargs))

        return SequentialModule(*layers)

    def _forward_impl(self, x, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpool(x)

        x = self.layer1(x, **kwargs)
        x = self.layer2(x, **kwargs)
        x = self.layer3(x, **kwargs)
        x = self.layer4(x, **kwargs)
        
        
        x = self.avgpool(x)
        
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def forward(self, x, **kwargs):
        return self._forward_impl(x, **kwargs)

@register_model
def birealnet18(pretrained=False, single_step_neuron: callable=None,**kwargs):
    """Constructs a BiRealNet-18 SNN model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], single_step_neuron=single_step_neuron, **kwargs)
    return model

@register_model
def birealnet34(single_step_neuron: callable=None, pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 SNN model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model
