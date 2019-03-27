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
from upsnet.nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from upsnet.rpn.generate_anchors import generate_anchors
from upsnet.bbox.bbox_transform import bbox_transform as bbox_pred, clip_boxes, bbox_overlaps
import numpy as np


class PyramidProposalFunction(Function):

    def __init__(self, feat_stride, scales, ratios, rpn_pre_nms_top_n, rpn_post_nms_top_n, threshold, rpn_min_size, individual_proposals=False, batch_idx=0, use_softnms=False, crowd_gt_roi=None):
        super(PyramidProposalFunction, self).__init__()
        self.feat_stride = feat_stride
        self.scales = np.array(scales)
        self.ratios = np.array(ratios)
        self.num_anchors = 3
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.threshold = threshold
        self.rpn_min_size = rpn_min_size
        self.individual_proposals = individual_proposals
        self.batch_idx = batch_idx
        self.use_softnms = use_softnms
        self.crowd_gt_roi = crowd_gt_roi

    def forward(self, cls_prob_p2, cls_prob_p3, cls_prob_p4, cls_prob_p5, cls_prob_p6,
                bbox_pred_p2, bbox_pred_p3, bbox_pred_p4, bbox_pred_p5, bbox_pred_p6, im_info):
        device_id = cls_prob_p2.get_device()
        nms = gpu_nms_wrapper(self.threshold, device_id=device_id) if not self.use_softnms else soft_nms_wrapper(self.threshold)
        context = torch.device('cuda', device_id)

        batch_size = cls_prob_p2.shape[0]
        if batch_size > 1:
            raise ValueError("Sorry, multiple images each device is not implemented")

        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        cls_probs = [cls_prob_p2, cls_prob_p3, cls_prob_p4, cls_prob_p5, cls_prob_p6]
        bbox_preds = [bbox_pred_p2, bbox_pred_p3, bbox_pred_p4, bbox_pred_p5, bbox_pred_p6]

        pre_nms_topN = self.rpn_pre_nms_top_n
        post_nms_topN = self.rpn_post_nms_top_n
        min_size = self.rpn_min_size

        proposal_list = []
        score_list = []
        im_info = im_info.numpy()

        for s in range(len(self.feat_stride)):
            stride = int(self.feat_stride[s])
            sub_anchors = generate_anchors(stride=stride, sizes=self.scales * stride, aspect_ratios=self.ratios)
            scores = cls_probs[s].cpu().numpy()
            bbox_deltas = bbox_preds[s].cpu().numpy()
            # 1. Generate proposals from bbox_deltas and shifted anchors
            # use real image size instead of padded feature map sizes
            height, width = scores.shape[-2:]

            # Enumerate all shifts
            shift_x = np.arange(0, width) * stride
            shift_y = np.arange(0, height) * stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

            # Enumerate all shifted anchors:
            #
            # add A anchors (1, A, 4) to
            # cell K shifts (K, 1, 4) to get
            # shift anchors (K, A, 4)
            # reshape to (K*A, 4) shifted anchors
            A = self.num_anchors
            K = shifts.shape[0]
            anchors = sub_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
            anchors = anchors.reshape((K * A, 4))
            # print(np.linalg.norm(anchors))

            # Transpose and reshape predicted bbox transformations to get them
            # into the same order as the anchors:
            #
            # bbox deltas will be (1, 4 * A, H, W) format
            # transpose to (1, H, W, 4 * A)
            # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
            # in slowest to fastest order
            # bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

            # Same story for the scores:
            #
            # scores are (1, A, H, W) format
            # transpose to (1, H, W, A)
            # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
            # scores = self._clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            if self.individual_proposals:
                # 4. sort all (proposal, score) pairs by score from highest to lowest
                # 5. take top pre_nms_topN (e.g. 6000)
                if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
                    order = np.argsort(-scores.squeeze())
                else:
                    # Avoid sorting possibly large arrays; First partition to get top K
                    # unsorted and then sort just those (~20x faster for 200k scores)
                    inds = np.argpartition(
                        -scores.squeeze(), pre_nms_topN
                    )[:pre_nms_topN]
                    order = np.argsort(-scores[inds].squeeze())
                    order = inds[order]
                    # order = np.argsort(-scores.squeeze())
                bbox_deltas = bbox_deltas[order, :]
                anchors = anchors[order, :]
                scores = scores[order]

            # Convert anchors into proposals via bbox transformations
            proposals = bbox_pred(anchors, bbox_deltas)

            # 2. clip predicted boxes to image
            proposals = clip_boxes(proposals, im_info[:2])

            # 3. remove predicted boxes with either height or width < threshold
            # (NOTE: convert min_size to input image scale stored in im_info[2])
            keep = self._filter_boxes(proposals, min_size * im_info[2])
            proposals = proposals[keep, :]
            scores = scores[keep]

            if self.crowd_gt_roi is not None:
                proposal_by_gt_overlap = bbox_overlaps(proposals, self.crowd_gt_roi * im_info[2])
                proposal_by_gt_overlap_max = proposal_by_gt_overlap.max(axis=1)
                keep = np.where(proposal_by_gt_overlap_max < 0.5)[0]
                proposals = proposals[keep, :]
                scores = scores[keep]

            if self.individual_proposals:
                # 6. apply nms (e.g. threshold = 0.7)
                # 7. take after_nms_topN (e.g. 300)
                # 8. return the top proposals (-> RoIs top)
                if self.use_softnms:
                    det, keep = nms(np.hstack((proposals, scores)).astype(np.float32))
                    det = det[keep]
                    det = det[np.argsort(det[:, 4])[::-1]]
                    if post_nms_topN > 0:
                        det = det[:post_nms_topN]
                    proposals = det[:, :4]
                    scores = det[:, 4]
                else:
                    keep = nms(np.hstack((proposals, scores)).astype(np.float32))
                    if post_nms_topN > 0:
                        keep = keep[:post_nms_topN]
                    proposals = proposals[keep, :]
                    scores = scores[keep]

            proposal_list.append(proposals)
            score_list.append(scores)

        proposals = np.vstack(proposal_list)
        scores = np.vstack(score_list)


        if not self.individual_proposals:
            # 4. sort all (proposal, score) pairs by score from highest to lowest
            # 5. take top pre_nms_topN (e.g. 6000)
            order = scores.ravel().argsort()[::-1]
            if pre_nms_topN > 0:
                order = order[:pre_nms_topN]
            proposals = proposals[order, :]
            scores = scores[order]

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            if self.use_softnms:
                det, keep = nms(np.hstack((proposals, scores)).astype(np.float32))
                det = det[keep]
                det = det[np.argsort(det[:, 4])[::-1]]
                if post_nms_topN > 0:
                    det = det[:post_nms_topN]
                proposals = det[:, :4]
                scores = det[:, 4]
            else:
                det = np.hstack((proposals, scores)).astype(np.float32)
                keep = nms(det)
                if post_nms_topN > 0:
                    keep = keep[:post_nms_topN]
                # pad to ensure output size remains unchanged
                if len(keep) < post_nms_topN:
                    pad = np.random.choice(keep, size=post_nms_topN - len(keep))
                    keep = np.hstack((keep, pad))
                proposals = proposals[keep, :]
                scores = scores[keep]
        else:
            scores = scores.squeeze()

        # Output rois array
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.ones((proposals.shape[0], 1), dtype=np.float32) * self.batch_idx
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        return torch.tensor(blob, requires_grad=False).pin_memory().to(context, dtype=torch.float32, non_blocking=True), \
               torch.tensor(scores, requires_grad=False).pin_memory().to(context, dtype=torch.float32, non_blocking=True)

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
        Clip boxes of the pad area.
        :param tensor: [n, c, H, W]
        :param pad_shape: [h, w]
        :return: [n, c, h, w]
        """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor

    def backward(self, grad_output):
        return None, None, None, None, None, None, None, None, None, None, None
