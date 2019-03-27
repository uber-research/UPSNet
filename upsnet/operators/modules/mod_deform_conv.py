# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from deeplab.operators.functions.mod_deform_conv import ModDeformConvFunction


class ModDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModDeformConv, self).__init__()
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        assert out_channels % deformable_groups == 0, 'out_channels must be divisible by deformable groups'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = Parameter(torch.Tensor(
            self.out_channels, self.in_channels // self.groups, *self.kernel_size).cuda())
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels).cuda())
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data, offset_mask):
        offset_1, offset_2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_1, offset_2), dim=1)
        mask = torch.sigmoid(mask) * 2

        return ModDeformConvFunction.apply(data, offset, mask, self.weight, self.bias, self.in_channels, self.out_channels,
                                           self.kernel_size, self.stride, self.padding, self.dilation, self.groups,
                                           self.deformable_groups)


class ModDeformConvWithOffsetMask(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModDeformConvWithOffsetMask, self).__init__()
        self.conv_offset_mask = nn.Conv2d(in_channels, kernel_size * kernel_size * 3 * deformable_groups, kernel_size=3, stride=1, padding=1)
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()
        self.conv = ModDeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

    def forward(self, x):
        return self.conv(x, self.conv_offset_mask(x))