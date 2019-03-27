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
# Written by Yuwen Xiong, Rui Hu
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils.timer import timeit
import math
from upsnet.config.config import config


class MaskTerm(nn.Module):

    def __init__(self, num_seg_classes, box_scale=1/4.0, class_mapping=None):
        super(MaskTerm, self).__init__()
        self.class_mapping = dict(zip(range(1, config.dataset.num_classes), range(num_seg_classes - config.dataset.num_classes + 1, num_seg_classes))) if class_mapping is None else class_mapping
        self.num_seg_classes = num_seg_classes
        self.box_scale = box_scale

    def forward(self, masks, boxes, cls_indices, seg_score):
        """

        :param masks: [num_boxes x c x 28 x 28]
        :param boxes: [num_boxes x 5]
        :param cls_indices: [num_boxes x 1]
        :param seg_score: [1 x num_seg_classes x h x w]
        :return: mask_energy: [1 x num_boxes x h x w]
        """

        assert seg_score.shape[0] == 1, "only support batch size = 1"
        cls_indices = cls_indices.cpu().numpy()

        # remove first dim which indicate batch id
        boxes = boxes[:, 1:] * self.box_scale
        im_shape = seg_score.shape[2:]

        # [n x num_boxes x h x w]
        mask_energy = torch.zeros((seg_score.shape[0], masks.shape[0], seg_score.shape[2], seg_score.shape[3]), device=seg_score.device)

        for i in range(cls_indices.shape[0]):
            ref_box = boxes[i, :].long()
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = max(w, 1)
            h = max(h, 1)
            mask = F.upsample(masks[i, 0, :, :].view(1, 1, config.network.mask_size, config.network.mask_size), size=(h, w), mode='bilinear', align_corners=False)
            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_shape[1])
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_shape[0])
            mask_energy[0, i, y_0:y_1, x_0:x_1] = \
                mask[0, 0, (y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]
        return mask_energy


class SegTerm(nn.Module):

    def __init__(self, num_seg_classes, box_scale=1/4.0, class_mapping=None, thresh=0.3):
        super(SegTerm, self).__init__()
        self.class_mapping = dict(zip(range(1, config.dataset.num_classes), range(num_seg_classes - config.dataset.num_classes + 1, num_seg_classes))) if class_mapping is None else class_mapping
        self.num_seg_classes = num_seg_classes
        self.num_inst_classes = len(self.class_mapping)
        self.box_scale = box_scale

    def forward(self, cls_indices, seg_score, boxes):
        """
        :param cls_indices: [num_boxes x 1]
        :param seg_score: [1 x num_seg_classes x h x w]
        :return: seg_energy: [1 x (num_seg_classes - num_inst_classes + num_boxes) x h x w]
        """

        assert seg_score.shape[0] == 1, "only support batch size = 1"
        cls_indices = cls_indices.cpu().numpy()
        seg_energy = seg_score[[0], :-self.num_inst_classes, :, :]

        boxes = boxes.cpu().numpy()
        boxes = boxes[:, 1:] * self.box_scale

        if cls_indices.size == 0:
            return seg_energy, torch.ones_like(seg_energy[[0], [0], :, :]).view(1, 1, seg_energy.shape[2], seg_energy.shape[3]) * -10
        else:
            seg_inst_energy = torch.zeros((seg_score.shape[0], cls_indices.shape[0], seg_score.shape[2], seg_score.shape[3]), device=seg_score.device)
            for i in range(cls_indices.shape[0]):
                if cls_indices[i] == 0:
                    continue
                y0 = int(boxes[i][1])
                y1 = int(boxes[i][3].round()+1)
                x0 = int(boxes[i][0])
                x1 = int(boxes[i][2].round()+1)
                seg_inst_energy[0, i, y0: y1, x0: x1] = seg_score[0, self.class_mapping[cls_indices[i]], y0: y1, x0: x1]

            return seg_energy, seg_inst_energy

