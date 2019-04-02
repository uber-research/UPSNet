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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
import matplotlib
import matplotlib.pyplot
from upsnet.config.config import config
from upsnet.bbox.bbox_transform import bbox_transform, bbox_overlaps
import cv2

class MaskMatching(nn.Module):

    def __init__(self, num_seg_classes, enable_void, class_mapping=None):
        super(MaskMatching, self).__init__()
        self.class_mapping = dict(zip(range(1, config.dataset.num_classes), range(num_seg_classes - config.dataset.num_classes + 1, num_seg_classes))) if class_mapping is None else class_mapping
        self.num_seg_classes = num_seg_classes
        self.num_inst_classes = len(self.class_mapping)
        self.enable_void = enable_void

    def forward(self, gt_segs, gt_masks, keep_inds=None):
        """
        :param gt_segs: [1 x h x w]
        :param gt_masks: [num_gt_boxes x h x w]
        :param keep_inds: [num_kept_boxes x 1]
        :return: matched_gt: [1 x h x w]
        """

        matched_gt = torch.ones_like(gt_segs) * -1
        matched_gt = torch.where(gt_segs <= config.dataset.num_seg_classes - config.dataset.num_classes, gt_segs, matched_gt)
        matched_gt = torch.where(gt_segs >= 255, gt_segs, matched_gt)
        if keep_inds is not None:
            gt_masks = gt_masks[keep_inds]

        for i in range(gt_masks.shape[0]):
            matched_gt[gt_masks[[i], :, :] != 0] = i + self.num_seg_classes - self.num_inst_classes
        if keep_inds is not None:
            matched_gt[matched_gt == -1] = self.num_seg_classes - self.num_inst_classes + gt_masks.shape[0]
        else:
            matched_gt[matched_gt == -1] = 255

        return matched_gt


class PanopticGTGenerate(nn.Module):

    def __init__(self, num_seg_classes, enable_void, class_mapping=None):
        super(PanopticGTGenerate, self).__init__()
        self.class_mapping = dict(zip(range(1, config.dataset.num_classes), range(num_seg_classes - config.dataset.num_classes + 1, num_seg_classes))) if class_mapping is None else class_mapping
        self.num_seg_classes = num_seg_classes
        self.num_inst_classes = len(self.class_mapping)
        self.enable_void = enable_void

    def forward(self, rois, bbox_pred, cls_score, label, gt_rois, cls_idx, seg_gt, mask_gt, im_shape):
        
        rois = rois.data.cpu().numpy()
        bbox_pred = bbox_pred.data.cpu().numpy()
        cls_score = cls_score.data.cpu().numpy()
        cls_pred = np.argmax(cls_score, axis=1)
        label = label.data.cpu().numpy()
        gt_rois = gt_rois.cpu().numpy()

        rois = rois[:, 1:]

        bbox_overlap = bbox_overlaps(rois, gt_rois[:, 1:])  # #rois x #gt_rois
        max_bbox_overlap = np.argmax(bbox_overlap, axis=1)
        max_overlap = np.ones((gt_rois.shape[0]), dtype=np.int32) * -1

        matched_gt = torch.ones_like(seg_gt) * -1
        matched_gt = torch.where(seg_gt <= config.dataset.num_seg_classes - config.dataset.num_classes, seg_gt, matched_gt)
        matched_gt = torch.where(seg_gt >= 255, seg_gt, matched_gt)

        keep = np.ones((rois.shape[0]), dtype=np.int32)

        for i in range(rois.shape[0]):
            if bbox_overlap[i, max_bbox_overlap[i]] > 0.5:
                if max_overlap[max_bbox_overlap[i]] == -1:
                    max_overlap[max_bbox_overlap[i]] = i
                elif bbox_overlap[max_overlap[max_bbox_overlap[i]], max_bbox_overlap[i]] > bbox_overlap[i, max_bbox_overlap[i]]:
                    keep[i] = 0
                else: 
                    keep[max_overlap[max_bbox_overlap[i]]] = 0
                    max_overlap[max_bbox_overlap[i]] = i
            elif cls_pred[i] == 0 and label[i] == 0:
                keep[i] = 0

        rois = rois[keep != 0]
        rois = np.hstack((np.zeros((rois.shape[0], 1)), rois))
        label = label[keep != 0]

        keep = np.cumsum(keep)
        if keep[-1] == 0:
            print(max_overlap)
            print(max_bbox_overlap)
            print(cls_pred)
            assert keep[-1] != 0

        for i in range(max_overlap.shape[0]):
            if max_overlap[i] != -1:
                roi = np.round(rois[keep[max_overlap[i]] - 1] / 4)
                mask_gt_i = mask_gt[[i]]
                matched_gt[mask_gt_i != 0] = int(keep[max_overlap[i]] - 1 + self.num_seg_classes - self.num_inst_classes)

        if config.train.panoptic_box_keep_fraction < 1:
            matched_gt[matched_gt == -1] = self.num_seg_classes - self.num_inst_classes + rois.shape[0]
        else:
            matched_gt[matched_gt == -1] = 255

        return torch.from_numpy(rois).to(matched_gt.device), torch.from_numpy(label).to(matched_gt.device), matched_gt



