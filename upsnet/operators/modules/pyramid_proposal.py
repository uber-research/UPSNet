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
from torch.nn.modules.module import Module
from ..functions.pyramid_proposal import PyramidProposalFunction
from torch.autograd import Variable
import pickle
import numpy as np

class PyramidProposal(Module):
    def __init__(self, feat_stride, scales, ratios, rpn_pre_nms_top_n, rpn_post_nms_top_n, threshold, rpn_min_size, individual_proposals=False, use_softnms=False):
        super(PyramidProposal, self).__init__()
        self.feat_stride = feat_stride
        self.scales = scales
        self.ratios = ratios
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.threshold = threshold
        self.rpn_min_size = rpn_min_size
        self.individual_proposals = individual_proposals
        self.use_softnms = use_softnms

    def forward(self, cls_prob, bbox_pred, im_info, roidb=None):
        rois = []
        scores = []

        if roidb is not None:
            roidb = pickle.loads(roidb.astype(np.uint8).tobytes())


        for i in range(im_info.shape[0]):
            if roidb is not None:
                crowd_gt_roi = roidb[i]['boxes'][np.where((roidb[i]['gt_classes'] > 0) & (roidb[i]['is_crowd'] == 1))[0], :]
                if crowd_gt_roi.size == 0:
                    crowd_gt_roi = None
            else:
                crowd_gt_roi = None
            pyramid_proposal_function = PyramidProposalFunction(self.feat_stride, self.scales, self.ratios, self.rpn_pre_nms_top_n,
                                                                self.rpn_post_nms_top_n, self.threshold, self.rpn_min_size,
                                                                self.individual_proposals, i, self.use_softnms, crowd_gt_roi)
            rois_im_i, scores_im_i = \
                pyramid_proposal_function(cls_prob[0][[i], :, :, :], cls_prob[1][[i], :, :, :], cls_prob[2][[i], :, :, :],
                                          cls_prob[3][[i], :, :, :], cls_prob[4][[i], :, :, :], bbox_pred[0][[i], :, :, :],
                                          bbox_pred[1][[i], :, :, :], bbox_pred[2][[i], :, :, :],
                                          bbox_pred[3][[i], :, :, :], bbox_pred[4][[i], :, :, :], torch.from_numpy(im_info[i, :]))
            rois.append(rois_im_i)
            scores.append(scores_im_i)
        rois = torch.cat(rois, 0).data
        scores = torch.cat(scores, 0).data
        _, idx = torch.sort(-scores, 0)
        idx = idx[:self.rpn_post_nms_top_n]


        return rois[idx, :], scores[idx]
