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

import torch
from torch.autograd import Function
from .._ext.roi_align import roi_align_cuda


class RoIAlignFunction(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale, sampling_ratio=2):
        super(RoIAlignFunction, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = sampling_ratio
        self.feature_size = None

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.shape
        num_rois = rois.shape[0]

        if not features.is_cuda:
            raise Exception('not implemented')

        output = features.new().resize_(num_rois, num_channels, self.pooled_height, self.pooled_width).zero_()

        roi_align_cuda.roi_align_forward(self.pooled_height, self.pooled_width, self.sampling_ratio, self.spatial_scale,
                                    features, rois, output)
        self.feature_size = features.size()
        self.rois = rois
        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = grad_output.new().resize_(batch_size, num_channels, data_height, data_width).zero_()
        roi_align_cuda.roi_align_backward(self.pooled_height, self.pooled_width, self.sampling_ratio, self.spatial_scale,
                                     grad_output, self.rois, grad_input)

        return grad_input, None
