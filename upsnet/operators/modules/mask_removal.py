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
import numpy as np
import cv2
from upsnet.bbox.bbox_transform import expand_boxes
from upsnet.config.config import config

class MaskRemoval(nn.Module):

    def __init__(self, fraction_threshold=0.3):
        super(MaskRemoval, self).__init__()
        self.fraction_threshold = fraction_threshold

    def forward(self, mask_rois, cls_prob, mask_prob, cls_idx, im_shape):
        """

        :param mask_rois: [n x 4]
        :param cls_prob: [n x 1]
        :param mask_prob: [n x 28 x 28]
        :param im_shape: [2] (h x w)
        :return:
        """
        mask_logit_gpu = mask_prob
        mask_energy = mask_rois.new_zeros(1, mask_rois.size(0), im_shape[0], im_shape[1])
        frame_id = 0

        context = mask_rois.device
        mask_rois = mask_rois.detach().cpu().numpy()
        cls_prob = cls_prob.detach().cpu().numpy()
        mask_logit = mask_logit_gpu.detach().cpu().numpy()
        cls_idx = cls_idx.detach().cpu().numpy()

        mask_image = np.zeros((np.max(cls_idx),) + im_shape, dtype=np.uint8)

        sorted_inds = np.argsort(cls_prob)[::-1]
        mask_rois = mask_rois[sorted_inds]
        cls_prob = cls_prob[sorted_inds]
        mask_logit = mask_logit[sorted_inds]
        cls_idx = cls_idx[sorted_inds] - 1
        if len(cls_idx) == 1 and cls_idx[0] == -1:
            mask_energy = mask_logit_gpu.new_zeros(1, 1, im_shape[0], im_shape[1])
            return torch.from_numpy(np.array([0], dtype=np.int64)).pin_memory().to(context, non_blocking=True), mask_energy

        keep_inds = []
        ref_boxes = mask_rois.astype(np.int32)

        for i in range(sorted_inds.shape[0]):
            ref_box = ref_boxes[i, :].astype(np.int32)
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = max(w, 1)
            h = max(h, 1)
            logit = cv2.resize(mask_logit[i].squeeze(), (w, h))
            logit_tensor = torch.from_numpy(logit).cuda()
            mask = np.array(logit > 0, dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_shape[1])
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_shape[0])

            crop_mask = mask[(y_0 - ref_box[1]):(y_1 - ref_box[1]),
                             (x_0 - ref_box[0]):(x_1 - ref_box[0])]
            mask_sum = crop_mask.sum()

            mask_image_crop = mask_image[cls_idx[i]][y_0:y_1, x_0:x_1]
            if mask_sum == 0 or (np.logical_and(mask_image_crop >= 1, crop_mask == 1).sum() / mask_sum > self.fraction_threshold):
                continue
            keep_inds.append(sorted_inds[i])
            mask_image[cls_idx[i]][y_0:y_1, x_0:x_1] += crop_mask
            mask_energy[0, frame_id, y_0: y_1, x_0: x_1] = logit_tensor[(y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]
            frame_id += 1

        mask_energy = mask_energy[:, :len(keep_inds)]
        if len(keep_inds) == 0:
            mask_energy = mask_logit_gpu.new_zeros(1, 1, im_shape[0], im_shape[1])
            return torch.from_numpy(np.array([0], dtype=np.int64)).pin_memory().to(context, non_blocking=True), mask_energy
        return torch.from_numpy(np.array(keep_inds)).pin_memory().to(context, non_blocking=True), mask_energy
