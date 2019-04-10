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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from upsnet.operators.modules.fpn_roi_align import FPNRoIAlign
from upsnet.operators.modules.roialign import RoIAlign
from upsnet.operators.functions.roialign import RoIAlignFunction
from upsnet.operators.modules.view import View
from upsnet.config.config import config

if config.train.use_horovod and config.network.use_syncbn:
    from upsnet.operators.modules.distbatchnorm import BatchNorm2d



class MaskBranch(nn.Module):

    def __init__(self, num_classes, dim_in=256, dim_hidden=256, with_norm='none'):
        super(MaskBranch, self).__init__()
        self.roi_pooling = FPNRoIAlign(config.network.mask_size // 2, config.network.mask_size // 2,
                                        [1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32])
        conv = nn.Conv2d

        assert with_norm in ['batch_norm', 'group_norm', 'none']

        if with_norm == 'batch_norm':
            norm = BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm

        if with_norm != 'none':
            self.mask_conv1 = nn.Sequential(*[conv(dim_in, dim_hidden, 3, 1, 1, bias=False), norm(dim_hidden), nn.ReLU(inplace=True)])
            self.mask_conv2 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1, bias=False), norm(dim_hidden), nn.ReLU(inplace=True)])
            self.mask_conv3 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1, bias=False), norm(dim_hidden), nn.ReLU(inplace=True)])
            self.mask_conv4 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1, bias=False), norm(dim_hidden), nn.ReLU(inplace=True)])
            self.mask_deconv1 = nn.Sequential(*[nn.ConvTranspose2d(dim_hidden, dim_hidden, 2, 2, 0), nn.ReLU(inplace=True)])
        else:
            self.mask_conv1 = nn.Sequential(*[conv(dim_in, dim_hidden, 3, 1, 1), nn.ReLU(inplace=True)])
            self.mask_conv2 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1), nn.ReLU(inplace=True)])
            self.mask_conv3 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1), nn.ReLU(inplace=True)])
            self.mask_conv4 = nn.Sequential(*[conv(dim_hidden, dim_hidden, 3, 1, 1), nn.ReLU(inplace=True)])
            self.mask_deconv1 = nn.Sequential(*[nn.ConvTranspose2d(dim_hidden, dim_hidden, 2, 2, 0), nn.ReLU(inplace=True)])

        self.mask_score = nn.Conv2d(dim_hidden, num_classes, 1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, feat, rois):
        pool_feat = self.roi_pooling(feat, rois)
        mask_conv1 = self.mask_conv1(pool_feat)
        mask_conv2 = self.mask_conv2(mask_conv1)
        mask_conv3 = self.mask_conv3(mask_conv2)
        mask_conv4 = self.mask_conv4(mask_conv3)
        mask_deconv1 = self.mask_deconv1(mask_conv4)
        mask_score = self.mask_score(mask_deconv1)
        return mask_score

class RCNN(nn.Module):

    def __init__(self, num_classes, num_reg_classes, pool_size=7, dim_in=256, dim_hidden=1024, with_fpn_pooling=True, with_dpooling=False, with_adaptive_pooling=False, with_heavier_head=False, with_norm='none'):
        super(RCNN, self).__init__()
        self.with_fpn_pooling = with_fpn_pooling
        if self.with_fpn_pooling:
            self.roi_pooling = FPNRoIAlign(pool_size, pool_size, [1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32])
        else:
            self.roi_pooling = RoIAlign(pool_size, pool_size, [1.0 / 16])
        
        assert with_norm in ['batch_norm', 'group_norm', 'none']
        if with_norm == 'batch_norm':
            norm = BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm
        
        if with_norm != 'none':
            self.fc6 = nn.Sequential(*[nn.Linear((pool_size ** 2) * dim_in, dim_hidden), View(-1, dim_hidden, 1, 1), norm(dim_hidden), View(-1, dim_hidden), nn.ReLU(inplace=True)])
            self.fc7 = nn.Sequential(*[nn.Linear(dim_hidden, dim_hidden), nn.ReLU(inplace=True)])
        else:
            self.fc6 = nn.Sequential(*[nn.Linear((pool_size ** 2) * dim_in, dim_hidden), nn.ReLU(inplace=True)])
            self.fc7 = nn.Sequential(*[nn.Linear(dim_hidden, dim_hidden), nn.ReLU(inplace=True)])
        self.cls_score = nn.Linear(dim_hidden, num_classes)
        self.bbox_pred = nn.Linear(dim_hidden, num_reg_classes * 4)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.fill_(0)
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
        nn.init.normal_(self.cls_score.weight.data, 0, 0.01)
        self.cls_score.bias.data.fill_(0)
        nn.init.normal_(self.bbox_pred.weight.data, 0, 0.001)
        self.bbox_pred.bias.data.fill_(0)

    def forward(self, feat, rois):
        pool_feat = self.roi_pooling(feat, rois)

        rcnn_feat = pool_feat.view(pool_feat.size(0), -1)
        fc6 = self.fc6(rcnn_feat)
        fc7 = self.fc7(fc6)
        fc_feat = fc7

        cls_score = self.cls_score(fc_feat)
        bbox_pred = self.bbox_pred(fc_feat)
        return {
            'cls_score': cls_score,
            'bbox_pred': bbox_pred,
            'fc_feat': fc_feat
        }

class RCNNLoss(nn.Module):
    def __init__(self):
        super(RCNNLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bbox_loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, cls_score, bbox_pred, cls_label, bbox_target, bbox_weight):
        cls_loss = self.cls_loss(cls_score, cls_label)
        bbox_loss = self.bbox_loss(bbox_pred * bbox_weight, bbox_target * bbox_weight)
        return cls_loss, bbox_loss

class MaskRCNNLoss(nn.Module):
    def __init__(self, batch_size):
        super(MaskRCNNLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bbox_loss = self.smooth_l1_loss

    def rcnn_accuracy(self, cls_score, cls_label):
        _, cls_pred = torch.max(cls_score.data, 1, keepdim=True)
        ignore = (cls_label == -1).long().sum()
        correct = (cls_pred.view(-1) == cls_label.data).long().sum() - ignore
        total = (cls_label.view(-1).shape[0])  - ignore

        return correct.float() / total.float()

    def mask_loss(self, input, target, weight):
        binary_input = (input >= 0).float()
        loss = -input * (target - binary_input) + torch.log(1 + torch.exp(input - 2 * input * binary_input))
        loss = loss * weight
        return loss.sum()

    def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
        loss_box = (torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign
                    + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)) * bbox_outside_weights
        return loss_box.sum() / loss_box.shape[0]

    def forward(self, cls_score, bbox_pred, mask_score, cls_label, bbox_target, bbox_inside_weight, bbox_outside_weight, mask_target):
        cls_loss = self.cls_loss(cls_score, cls_label)
        bbox_loss = self.bbox_loss(bbox_pred, bbox_target, bbox_inside_weight, bbox_outside_weight)
        rcnn_acc = self.rcnn_accuracy(cls_score.detach(), cls_label)
        mask_target = mask_target.view(mask_target.shape[0], -1, config.network.mask_size, config.network.mask_size)
        mask_weight = mask_target != -1
        mask_loss = self.mask_loss(mask_score, mask_target, mask_weight.float()) / (mask_weight.float().sum() + 1e-10)

        return cls_loss, bbox_loss, mask_loss, rcnn_acc
