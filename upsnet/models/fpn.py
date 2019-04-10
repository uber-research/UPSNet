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
import torch.nn as nn
import torch.nn.functional as F

from upsnet.config.config import config

if config.train.use_horovod and config.network.use_syncbn:
    from upsnet.operators.modules.distbatchnorm import BatchNorm2d

class FPN(nn.Module):

    def __init__(self, feature_dim, with_extra_level=True, with_bottom_up_path_aggregation=False, with_norm='none', upsample_method='nearest'):
        super(FPN, self).__init__()
        self.feature_dim = feature_dim
        self.with_bottom_up_path_aggregation = with_bottom_up_path_aggregation
        assert upsample_method in ['nearest', 'bilinear']

        def interpolate(input):
            return F.interpolate(input, scale_factor=2, mode=upsample_method, align_corners=False if upsample_method == 'bilinear' else None)
        self.fpn_upsample = interpolate

        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_extra_level:
            self.fpn_p6 = nn.MaxPool2d(kernel_size=1, stride=2)
        if config.network.fpn_with_gap:
            self.fpn_gap = nn.Linear(2048, feature_dim)
        
        if with_norm == 'batch_norm':
            norm = BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm

        if with_norm != 'none':
            self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(2048, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(1024, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(512, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p2_1x1 = nn.Sequential(*[nn.Conv2d(256, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p5 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p4 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p3 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p2 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
        else:
            self.fpn_p5_1x1 = nn.Conv2d(2048, feature_dim, 1)
            self.fpn_p4_1x1 = nn.Conv2d(1024, feature_dim, 1)
            self.fpn_p3_1x1 = nn.Conv2d(512, feature_dim, 1)
            self.fpn_p2_1x1 = nn.Conv2d(256, feature_dim, 1)
            self.fpn_p5 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p4 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p3 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p2 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, res2, res3, res4, res5):
        fpn_p5_1x1 = self.fpn_p5_1x1(res5)
        fpn_p4_1x1 = self.fpn_p4_1x1(res4)
        fpn_p3_1x1 = self.fpn_p3_1x1(res3)
        fpn_p2_1x1 = self.fpn_p2_1x1(res2)

        if config.network.fpn_with_gap:
            fpn_gap = self.fpn_gap(F.adaptive_avg_pool2d(res5, (1, 1)).squeeze()).view(-1, self.feature_dim, 1, 1)
            fpn_p5_1x1 = fpn_p5_1x1 + fpn_gap

        fpn_p5_upsample = self.fpn_upsample(fpn_p5_1x1)
        fpn_p4_plus = fpn_p5_upsample + fpn_p4_1x1
        fpn_p4_upsample = self.fpn_upsample(fpn_p4_plus)
        fpn_p3_plus = fpn_p4_upsample + fpn_p3_1x1
        fpn_p3_upsample = self.fpn_upsample(fpn_p3_plus)
        fpn_p2_plus = fpn_p3_upsample + fpn_p2_1x1

        fpn_p5 = self.fpn_p5(fpn_p5_1x1)
        fpn_p4 = self.fpn_p4(fpn_p4_plus)
        fpn_p3 = self.fpn_p3(fpn_p3_plus)
        fpn_p2 = self.fpn_p2(fpn_p2_plus)

        if hasattr(self, 'fpn_p6'):
            fpn_p6 = self.fpn_p6(fpn_p5)
            return fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6
        else:
            return fpn_p2, fpn_p3, fpn_p4, fpn_p5
