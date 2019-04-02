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
import numpy as np
import torch
from torch.autograd import Function, Variable
from .._ext.deform_conv import deform_conv_cuda

class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, data, offset, weight, bias, in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, deformable_groups):
        if weight.requires_grad or data.requires_grad:
            ctx.save_for_backward(data, offset, weight, bias)
        ctx.in_channels         = in_channels
        ctx.out_channels        = out_channels
        ctx.kernel_size         = kernel_size
        ctx.stride              = stride
        ctx.padding             = padding
        ctx.dilation            = dilation
        ctx.groups              = groups
        ctx.deformable_groups   = deformable_groups
        ctx.data_shape          = tuple(data.size())
        ctx.offset_shape        = tuple(offset.size())
        if not data.is_cuda or not offset.is_cuda or not weight.is_cuda or (bias is not None and not bias.is_cuda):
            raise Exception('not implemented')

        DeformConvFunction.shape_setup(ctx)
        col_buffer = data.new().resize_(int(ctx.in_channels * np.prod(ctx.kernel_size)), ctx.output_shape[2], ctx.output_shape[3]).zero_()
        output = data.new().resize_(ctx.output_shape[0], ctx.output_shape[1], ctx.output_shape[2], ctx.output_shape[3]).zero_()

        for i in range(ctx.data_shape[0]):
            deform_conv_cuda.deform_im2col(data[i, :, :, :], offset[i, :, :, :], ctx.data_shape,
                                           tuple(col_buffer.size()), ctx.kernel_size, ctx.padding, ctx.stride, ctx.dilation,
                                           1, ctx.deformable_groups, col_buffer)
            output[i, :, :, :] =\
                torch.mm(weight.view(-1, int(ctx.in_channels * np.prod(ctx.kernel_size))),
                         col_buffer.view(int(ctx.in_channels * np.prod(ctx.kernel_size)), -1))\
                     .view(ctx.output_shape[1], ctx.output_shape[2], ctx.output_shape[3])
        if bias is not None:
            output += bias.view(1, bias.size(0), 1, 1).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        data, offset, weight, bias = ctx.saved_tensors
        if not grad_output.is_cuda:
            raise Exception('not implemented')
        grad_data = data.new().resize_as_(data).zero_()
        grad_offset = offset.new().resize_as_(offset).zero_()
        grad_weight = weight.new().resize_as_(weight).zero_()


        for i in range(ctx.data_shape[0]):
            col_buffer = torch.mm(weight.view(-1, int(ctx.in_channels * np.prod(ctx.kernel_size))).t(),
                                  grad_output.data[i, :, :, :].view(ctx.out_channels, -1))\
                              .view((-1, grad_output.size(2), grad_output.size(3)))

            deform_conv_cuda.deform_col2im_coord(col_buffer, data[i, :, :, :], offset[i, :, :, :], ctx.data_shape,
                                                 tuple(col_buffer.size()), ctx.kernel_size, ctx.padding, ctx.stride, ctx.dilation,
                                                 1, ctx.deformable_groups, grad_offset[i, :, :, :])

            deform_conv_cuda.deform_col2im(col_buffer, offset[i, :, :, :], ctx.data_shape,
                                           tuple(col_buffer.size()), ctx.kernel_size, ctx.padding, ctx.stride, ctx.dilation,
                                           1, ctx.deformable_groups, grad_data[i, :, :, :])

            deform_conv_cuda.deform_im2col(data[i, :, :, :], offset[i, :, :, :], ctx.data_shape,
                                           tuple(col_buffer.size()), ctx.kernel_size, ctx.padding, ctx.stride, ctx.dilation,
                                           1, ctx.deformable_groups, col_buffer)

            grad_weight += torch.mm(grad_output.data[i, :, :, :].view(grad_output.size(1), -1),
                                    col_buffer.view(col_buffer.size(0), -1).t()).view_as(grad_weight)

        if bias is not None:
            grad_bias = grad_output.data.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True).squeeze()
        else:
            grad_bias = None
        return Variable(grad_data), Variable(grad_offset), Variable(grad_weight), Variable(grad_bias) if grad_bias is not None else bias, \
               None, None, None, None, None, None, None, None

    @staticmethod
    def shape_setup(ctx):
        ctx.kernel_dim = ctx.in_channels / ctx.groups * np.prod(ctx.kernel_size)

        ctx.output_shape = \
            (ctx.data_shape[0], ctx.out_channels,
             (ctx.data_shape[2] + 2 * ctx.padding[0] - ctx.dilation[0] * (ctx.kernel_size[0] - 1) - 1) // ctx.stride[0] + 1,
             (ctx.data_shape[3] + 2 * ctx.padding[1] - ctx.dilation[1] * (ctx.kernel_size[1] - 1) - 1) // ctx.stride[1] + 1)

        ctx.input_dim = np.prod(ctx.data_shape[1:])
        ctx.input_offset_dim = np.prod(ctx.offset_shape[1:])




