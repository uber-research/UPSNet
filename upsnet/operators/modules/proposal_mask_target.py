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
from torch.autograd import Variable
from torch.nn.modules.module import Module
from ..functions.proposal_mask_target import ProposalMaskTargetFunction
from upsnet.dataset.json_dataset import add_proposals
from upsnet.bbox.sample_rois import sample_rois
from collections import defaultdict
import time
import pickle

class ProposalMaskTarget(Module):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction, mask_size, binary_thresh):
        super(ProposalMaskTarget, self).__init__()
        self.num_classes = num_classes
        self.batch_images = batch_images
        self.batch_rois = batch_rois
        self.fg_fraction = fg_fraction
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

    def forward(self, rois, roidb, im_info):

        context = torch.device('cuda', rois.get_device())

        assert self.batch_rois == -1 or self.batch_rois % self.batch_images == 0, \
            'batchimages {} must devide batch_rois {}'.format(self.batch_images, self.batch_rois)
        all_rois = rois.data.cpu().numpy()
        add_proposals([roidb], all_rois, im_info[:, 2], crowd_thresh=0)

        blobs = defaultdict(list)

        for im_i, entry in enumerate([roidb]):
            frcn_blobs = sample_rois(entry, im_info[im_i, 2], im_i)
            for k, v in frcn_blobs.items():
                blobs[k].append(v)

        return torch.cat([torch.tensor(_, dtype=torch.float32, requires_grad=False).pin_memory().to(context, non_blocking=True) for _ in blobs['rois']], 0),\
               torch.cat([torch.tensor(_, dtype=torch.int64, requires_grad=False).pin_memory().to(context, non_blocking=True) for _ in blobs['labels_int32']], 0),\
               torch.cat([torch.tensor(_, dtype=torch.float32, requires_grad=False).pin_memory().to(context, non_blocking=True) for _ in blobs['bbox_targets']], 0),\
               torch.cat([torch.tensor(_, dtype=torch.float32, requires_grad=False).pin_memory().to(context, non_blocking=True) for _ in blobs['bbox_inside_weights']], 0),\
               torch.cat([torch.tensor(_, dtype=torch.float32, requires_grad=False).pin_memory().to(context, non_blocking=True) for _ in blobs['bbox_outside_weights']], 0),\
               torch.cat([torch.tensor(_, dtype=torch.float32, requires_grad=False).pin_memory().to(context, non_blocking=True) for _ in blobs['mask_rois']], 0),\
               torch.cat([torch.tensor(_, dtype=torch.float32, requires_grad=False).pin_memory().to(context, non_blocking=True) for _ in blobs['mask_int32']], 0), \
               torch.cat([torch.tensor(_, dtype=torch.uint8, requires_grad=False).pin_memory().to(context, non_blocking=True) for _ in blobs['roi_has_mask_int32']], 0), \
               torch.cat([torch.tensor(_, dtype=torch.int64, requires_grad=False).pin_memory().to(context, non_blocking=True) for _ in blobs['nongt_inds']], 0)
