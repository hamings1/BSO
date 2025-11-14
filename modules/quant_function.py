import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Function


class QuantConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        w = self.weight 
        output = F.conv2d(input, w, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output