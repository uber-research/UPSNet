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
from torch.nn.modules.module import Module
from torch.autograd import Variable
from upsnet.bbox.bbox_transform import bbox_transform as bbox_pred, clip_boxes
from upsnet.config.config import config
from upsnet.nms.nms import py_nms_wrapper, gpu_nms_wrapper, cpu_nms_wrapper

class MaskROI(Module):
    def __init__(self, clip_boxes, bbox_class_agnostic, top_n, num_classes, nms_thresh=None, class_agnostic=False, score_thresh=None):
        super(MaskROI, self).__init__()
        self.clip_boxes = clip_boxes
        self.bbox_class_agnostic = bbox_class_agnostic
        self.top_n = top_n
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh if nms_thresh is not None else config.test.nms_thresh
        self.class_agnostic = class_agnostic
        self.nms_classes = num_classes if not class_agnostic else 2
        self.score_thresh = score_thresh if score_thresh is not None else config.test.score_thresh

    def forward(self, bottom_rois, bbox_delta, cls_prob, im_info, nms=True, cls_score=None, cls_label=None):

        context = bottom_rois.get_device()
        nms = gpu_nms_wrapper(self.nms_thresh, device_id=context)


        bottom_rois     = bottom_rois.cpu().detach().numpy()
        bbox_delta      = bbox_delta.cpu().detach().numpy()
        cls_prob        = cls_prob.cpu()
        cls_prob_np = cls_prob.detach().numpy()
        if cls_score is not None:
            cls_prob = cls_score.cpu()
        if cls_label is not None:
            cls_label = cls_label.cpu()

        proposal = bbox_pred(bottom_rois[:, 1:], bbox_delta, config.network.bbox_reg_weights)

        if self.clip_boxes:
            proposal = clip_boxes(proposal, im_info[0, :2])

        cls_idx = [[_ for __ in range(proposal.shape[0])] for _ in range(self.num_classes)]


        if self.class_agnostic:
            cls_prob_np = cls_prob_np[:, 1:].reshape((-1, 1))
            # n x 1 -> n x 2
            cls_prob_np = np.hstack((np.zeros_like(cls_prob_np), cls_prob_np))
            cls_prob = cls_prob[:, 1:].contiguous().view(-1, 1)
            # n x 1 -> n x 2
            cls_prob = torch.cat([torch.zeros_like(cls_prob), cls_prob], dim=1)

            proposal = proposal.reshape((proposal.shape[0], -1, 4))[:, 1:, :].reshape((-1, 4))
            # n x 4 -> n x 8
            proposal = np.hstack((np.zeros_like(proposal), proposal))

            cls_idx = np.array(cls_idx).T[:, 1:].reshape((1, -1))
            # 1 x n -> 2 x n
            cls_idx = np.vstack((np.zeros_like(cls_idx), cls_idx)).tolist()

            if cls_label  is not None:
                cls_label = cls_label.view(-1, 1).expand(-1, self.num_classes).contiguous().view(-1, 1)

        cls_boxes = [[] for _ in range(self.nms_classes)]
        scores_th = [[] for _ in range(self.nms_classes)]
        if cls_score is not None:
            feat_th = [[] for _ in range(self.nms_classes)]
        if cls_label is not None:
            cls_label_th = [[] for _ in range(self.nms_classes)]
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class

        for j in range(1, self.nms_classes):
            inds = np.where(cls_prob_np[:, j] > self.score_thresh)[0]
            scores_j = cls_prob_np[inds, j]
            boxes_j = proposal[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32)

            # nms
            keep = [] if len(dets_j) == 0 else nms(dets_j)
            nms_dets = dets_j[keep, :]
            scores_th[j] = cls_prob[torch.from_numpy(inds).long(), j][torch.from_numpy(np.array(keep)).long()]

            # Refine the post-NMS boxes using bounding-box voting
            cls_boxes[j] = nms_dets
            cls_idx[j] = np.array(cls_idx[j])[inds][keep]
            if cls_label is not None:
                cls_label_th[j] = cls_label[inds][keep].numpy()

        # Limit to max_per_image detections **over all classes**
        if config.test.max_det > 0:
            image_scores = np.hstack(
                [cls_boxes[j][:, -1] for j in range(1, self.nms_classes)]
            )

            if len(image_scores) > config.test.max_det:
                image_thresh = np.sort(image_scores)[-config.test.max_det]
                for j in range(1, self.nms_classes):
                    keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                    cls_boxes[j] = cls_boxes[j][keep, :]
                    scores_th[j] = scores_th[j][torch.from_numpy(np.array(keep))]
                    cls_idx[j] = cls_idx[j][keep]
                    if cls_score is not None:
                        feat_th[j] = feat_th[j][torch.from_numpy(np.array(keep))]
                    if cls_label is not None:
                        cls_label_th[j] = cls_label_th[j][keep]

        im_results = np.vstack([cls_boxes[j] for j in range(1, self.nms_classes)])
        boxes = np.zeros(im_results.shape)
        boxes[:,1:] = im_results[:, :-1]
        scores = im_results[:, -1]
        cls_idx = np.hstack((cls_idx[1:]))
        if cls_label is not None:
            cls_label = np.hstack((cls_label_th[1:]))
            scores_th = torch.cat(scores_th[1:]).to(context, non_blocking=True)
            return scores_th, \
                   torch.from_numpy(boxes).float().pin_memory().to(context, non_blocking=True), \
                   torch.from_numpy(cls_idx).long().pin_memory().to(context, non_blocking=True), \
                   cls_label

        if scores.size == 0:
            scores = np.ones((1,))
            boxes = np.zeros((1, 5))
            cls_idx = np.zeros((1,))
            return torch.from_numpy(scores).float().to(context, non_blocking=True), \
                   torch.from_numpy(boxes).float().to(context, non_blocking=True), \
                   torch.from_numpy(cls_idx).long().to(context, non_blocking=True)
        else:
            scores_th = torch.cat(scores_th[1:]).to(context, non_blocking=True)
            return scores_th, \
                   torch.from_numpy(boxes).float().pin_memory().to(context, non_blocking=True), \
                   torch.from_numpy(cls_idx).long().pin_memory().to(context, non_blocking=True)

    @staticmethod
    def bbox_transform(bbox, bbox_delta, im_info=None):
        """

        :param bbox: [num_boxes, 4]
        :param bbox_delta: [num_boxes, (4 * num_reg_classes - 1)]
        :param im_info: [1, 3] -> [[height, width, scale]]
        :return: transformed_bbox: [num_boxes, 4, num_reg_classes]
        """

        xmin, ymin, xmax, ymax = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

        bbox_width = xmax - xmin + 1.0
        bbox_height = ymax - ymin + 1.0
        center_x = xmin + 0.5 * bbox_width
        center_y = ymin + 0.5 * bbox_height

        dx, dy, dw, dh = bbox_delta[:, 0::4],bbox_delta[:, 1::4], bbox_delta[:, 2::4], bbox_delta[:, 3::4]

        dx = dx / config.network.bbox_reg_weights[0]
        dy = dy / config.network.bbox_reg_weights[1]
        dw = dw / config.network.bbox_reg_weights[2]
        dh = dh / config.network.bbox_reg_weights[3]

        transformed_center_x = center_x.view(-1, 1) + bbox_width.view(-1, 1) * dx
        transformed_center_y = center_y.view(-1, 1) + bbox_height.view(-1, 1) * dy
        transformed_width = bbox_width.view(-1, 1) * dw.exp()
        transformed_height = bbox_height.view(-1, 1) * dh.exp()

        w_offset = 0.5 * transformed_width
        h_offset = 0.5 * transformed_height
        transformed_xmin = transformed_center_x - w_offset
        transformed_ymin = transformed_center_y - h_offset
        transformed_xmax = transformed_center_x + w_offset - 1
        transformed_ymax = transformed_center_y + h_offset - 1

        # transformed_bbox = torch.cat([transformed_xmin, transformed_ymin, transformed_xmax, transformed_ymax], dim=1)
        transformed_bbox = torch.zeros_like(bbox_delta)
        transformed_bbox[:, 0::4] = transformed_xmin
        transformed_bbox[:, 1::4] = transformed_ymin
        transformed_bbox[:, 2::4] = transformed_xmax
        transformed_bbox[:, 3::4] = transformed_ymax
        # [num_boxes, num_reg_classes - 1, 4]
        transformed_bbox = transformed_bbox.view((transformed_bbox.shape[0], -1, 4))


        if im_info is not None:
            # [1, 2]
            im_wh = torch.from_numpy(im_info[0, [1, 0]] - 1.0).to(transformed_bbox.device, non_blocking=True)
            # [1, 1, 4]
            im_wh = im_wh.repeat(1, 2).unsqueeze(0)
            transformed_bbox = torch.min(transformed_bbox, im_wh)
            transformed_bbox = torch.max(transformed_bbox, torch.zeros_like(transformed_bbox))

        return transformed_bbox.transpose(1, 2)
