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
from ..functions.roialign import RoIAlignFunction

class FPNRoIAlign(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, with_expand=False):
        super(FPNRoIAlign, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = spatial_scale
        self.with_expand = with_expand
        self.roi_pooling = RoIAlignFunction

    def forward(self, feat, rois):

        context = torch.device('cuda', feat[0].get_device())
        rois = rois.detach().cpu().numpy()
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        feat_id = np.clip(np.floor(2 + np.log2(np.sqrt(w * h) / 224 + 1e-6)), 0, 3)
        feat_no = []
        rois_fpn = []

        for i in range(4):
            feat_idx = np.where(feat_id == i)[0]
            if len(feat_idx) == 0:
                rois_fpn.append(np.zeros((1, 5)))
                feat_no.append(-1)
            else:
                rois_fpn.append(rois[feat_idx])
                feat_no.append(feat_idx)

        rois_index = torch.tensor(np.argsort(np.hstack(feat_no))[-rois.shape[0]:], dtype=torch.int64, requires_grad=False).to(context)


        pool_feat = []
        for i in range(len(self.spatial_scale)):
            rois_fpn[i] = torch.tensor(rois_fpn[i], dtype=torch.float32, requires_grad=False).to(context)
            pool_feat.append(
                self.roi_pooling(self.pooled_height, self.pooled_width, self.spatial_scale[i])(feat[i], rois_fpn[i]))

        pool_feat = torch.cat(pool_feat, dim=0)
        pool_feat = torch.index_select(pool_feat, 0, rois_index)
        return pool_feat

