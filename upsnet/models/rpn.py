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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from upsnet.config.config import config
if config.train.use_horovod and config.network.use_syncbn:
    from upsnet.operators.modules.distbatchnorm import BatchNorm2d


class RPN(nn.Module):
    def __init__(self, num_anchors=15, input_dim=256, with_norm='none'):
        super(RPN, self).__init__()
        self.num_anchors = num_anchors
        if with_norm == 'batch_norm':
            norm = BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm

        if with_norm != 'none':
            self.conv_proposal = nn.Sequential(*[nn.Conv2d(input_dim, input_dim, 3, padding=1), norm(input_dim), nn.ReLU(inplace=True)])
        else:
            self.conv_proposal = nn.Sequential(*[nn.Conv2d(input_dim, input_dim, 3, padding=1), nn.ReLU(inplace=True)])
        self.cls_score = nn.Conv2d(input_dim, self.num_anchors, 1)
        self.bbox_pred = nn.Conv2d(input_dim, self.num_anchors * 4, 1)
        self.initialize()

    def initialize(self):
        for m in [self.conv_proposal[0], self.cls_score, self.bbox_pred]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, data):
        x = self.conv_proposal(data)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        cls_prob = F.sigmoid(cls_score)
        return cls_score, bbox_pred, cls_prob


class RPNLoss(nn.Module):

    def __init__(self, rpn_batch_size, with_fpn=True):
        super(RPNLoss, self).__init__()
        self.rpn_cls_loss = F.binary_cross_entropy_with_logits
        self.rpn_bbox_loss = self.smooth_l1_loss
        self.rpn_batch_size = rpn_batch_size
        self.with_fpn = True

    def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=3.0):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().type(bbox_pred.dtype)
        loss_box = (torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign
                    + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)) * bbox_outside_weights
        return loss_box.sum()

    def forward(self, rpn_cls_score, rpn_bbox_pred, label):

        if self.with_fpn:
            rpn_cls_loss, rpn_bbox_loss = [], []
            for cls_score, bbox_pred, stride in zip(rpn_cls_score, rpn_bbox_pred, [4, 8, 16, 32, 64]):
                rpn_labels = label['rpn_labels_fpn{}'.format(stride)][:, :, :cls_score.size(2), :cls_score.size(3)]
                bbox_targets = label['rpn_bbox_targets_fpn{}'.format(stride)][:, :, :bbox_pred.size(2), :bbox_pred.size(3)]
                bbox_inside_weights = label['rpn_bbox_inside_weights_fpn{}'.format(stride)][:, :, :bbox_pred.size(2), :bbox_pred.size(3)]
                bbox_outside_weights = label['rpn_bbox_outside_weights_fpn{}'.format(stride)][:, :, :bbox_pred.size(2), :bbox_pred.size(3)]

                rpn_cls_loss.append(self.rpn_cls_loss(cls_score, rpn_labels.type(cls_score.dtype), (rpn_labels != -1).type(cls_score.dtype), reduction='sum') / self.rpn_batch_size)
                rpn_bbox_loss.append(self.rpn_bbox_loss(bbox_pred, bbox_targets.type(bbox_pred.dtype), bbox_inside_weights.type(bbox_pred.dtype), bbox_outside_weights.type(bbox_pred.dtype)) / bbox_pred.shape[0])
            rpn_cls_loss_sum, rpn_bbox_loss_sum = reduce(lambda x, y: x + y, rpn_cls_loss), reduce(lambda x, y: x + y, rpn_bbox_loss)
            return rpn_cls_loss_sum, rpn_bbox_loss_sum
        
        else:
            rpn_labels = label['rpn_labels'][:, :, :cls_score.size(2), :cls_score.size(3)]
            bbox_targets = label['rpn_bbox_targets'][:, :, :bbox_pred.size(2), :bbox_pred.size(3)]
            bbox_inside_weights = label['rpn_bbox_inside_weights'][:, :, :bbox_pred.size(2), :bbox_pred.size(3)]
            bbox_outside_weights = label['rpn_bbox_outside_weights'][:, :, :bbox_pred.size(2), :bbox_pred.size(3)]
            rpn_cls_loss = self.rpn_cls_loss(rpn_cls_score, rpn_labels.type(rpn_cls_score.dtype), (rpn_labels != -1).type(rpn_cls_score.dtype), reduction='sum') / self.rpn_batch_size
            rpn_bbox_loss = self.rpn_bbox_loss(rpn_bbox_pred, bbox_targets.type(rpn_bbox_pred.dtype), bbox_inside_weights.type(rpn_bbox_pred.dtype), bbox_outside_weights.tyep(rpn_bbox_pred.dtype)) / rpn_bbox_pred.shape[0]
            return rpn_cls_loss, rpn_bbox_loss
