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

from torch.nn.modules.module import Module
from ..functions.roialign import RoIAlignFunction


class RoIAlign(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)
