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
# --------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()
config.debug_mode = False
# config.output_path = ''
# config.model_prefix = ''
# config.symbol = ''
# config.gpus = ''

# network related params
config.network = edict()
# config.network.pretrained = ''
config.network.backbone_fix_bn = True
config.network.backbone_with_dilation = False
config.network.backbone_with_dpyramid = False
config.network.backbone_with_dconv = 100
config.network.backbone_freeze_at = 2
config.network.use_caffe_model = True
config.network.use_syncbn = False

config.network.has_rcnn = True
config.network.has_mask_head = True
config.network.has_fcn_head = False
config.network.has_panoptic_head = False

config.network.pixel_means = np.array((102.9801, 115.9465, 122.7717,))

config.network.cls_agnostic_bbox_reg = False
config.network.rcnn_feat_stride = 32
config.network.bbox_reg_weights = (10., 10., 5., 5.,)

# rpn
config.network.rpn_feat_stride = (4, 8, 16, 32, 64,)
config.network.anchor_ratios = (0.5, 1, 2)
config.network.anchor_scales = (8,)
config.network.num_anchors = 3
config.network.rpn_with_norm = 'none'

# fpn
config.network.has_fpn = True
config.network.fpn_feature_dim = 256
config.network.fpn_with_gap = False
config.network.fpn_upsample_method = 'nearest'
config.network.fpn_with_norm = 'none'

# rcnn
config.network.rcnn_with_norm = 'none'

# mask rcnn
config.network.mask_size = 28
config.network.binary_thresh = 0.5
config.network.has_mask_rcnn = True 

# fcn
config.network.fcn_with_norm = 'none'
config.network.fcn_num_layers = 3

# dataset related params
config.dataset = edict()
# config.dataset.dataset = ''
# config.dataset.image_set = ''
# config.dataset.test_image_set = ''
# config.dataset.root_path = ''
# config.dataset.dataset_path = ''
# config.dataset.num_classes = 0
# config.dataset.num_seg_classes = 0

# training related params
config.train = edict()

config.train.use_horovod = False
config.train.lr_schedule = 'step'
config.train.flip = True
config.train.shuffle = True
config.train.resume = False
config.train.begin_iteration = 0
config.train.eval_data = True

# config.train.warmup_iteration = 0
# config.train.lr = 0
# config.train.wd = 0
# config.train.momentum = 0
# config.train.batch_size = 0

# panoptic head related param
# config.train.fcn_loss_weight = 0
# config.train.fcn_with_roi_loss = False
# config.train.panoptic_loss_weight = 0
# config.train.panoptic_boox_keep_fraction = 0

# RCNN
config.train.batch_rois = 512
config.train.fg_fraction = 0.25
config.train.fg_thresh = 0.5
config.train.bg_thresh_hi = 0.5
config.train.bg_thresh_lo = 0.0
config.train.bbox_regression_thresh = 0.5
config.train.bbox_weights = np.array((1.0, 1.0, 1.0, 1.0,))

# RPN
config.train.rpn_batch_size = 256
config.train.rpn_fg_fraction = 0.5
config.train.rpn_positive_overlap = 0.7
config.train.rpn_negative_overlap = 0.3
config.train.rpn_clobber_positive = False
config.train.rpn_individual_proposals = True
config.train.rpn_straddle_thresh = 0
config.train.rpn_bbox_weights = (1.0, 1.0, 1.0, 1.0,)
config.train.rpn_positive_weight = -1.0

# RPN proposal
config.train.rpn_nms_thresh = 0.7
config.train.rpn_pre_nms_top_n = 2000
config.train.rpn_post_nms_top_n = 2000
config.train.rpn_min_size = 0
config.train.bbox_normalization_precomputed = True
config.train.bbox_means = (.0, .0, .0, .0,)
config.train.bbox_stds = (.1, .1, .2, .2,)
config.train.crowd_filter_thresh = 0.7
config.train.gt_min_area = -1
config.train.bbox_thresh = 0.5

config.train.fcn_with_roi_loss = False
config.train.fcn_with_negative_loss = False
config.train.fcn_use_focal_loss = False
config.train.fcn_focal_loss_gamma = 1.0
config.train.panoptic_with_roi_loss = False
config.train.panoptic_use_focal_loss = False
config.train.panoptic_focal_loss_gamma = 1.0

config.train.bbox_loss_weight = 1.
config.train.fcn_loss_weight = 1.
config.train.panoptic_loss_weight = 1.


config.test = edict()

config.test.vis_mask = False
# config.test.scales = [0]
# config.test.max_size = 0
# config.test.batch_size = 0
# config.test.test_iteration = 0

# RPN proposal
config.test.rpn_nms_thresh = 0.7
config.test.rpn_pre_nms_top_n = 1000
config.test.rpn_post_nms_top_n = 1000
config.test.rpn_min_size = 0

# RCNN
config.test.nms_thresh = 0.5
config.test.max_det = 100
config.test.score_thresh = 0.05

# Panoptic
config.test.panoptic_score_thresh = 0.6
config.test.panoptic_stuff_area_limit = 4096


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'train':
                        if 'bbox_weights' in v:
                            v['bbox_weights'] = np.array(v['bbox_weights'])
                    elif k == 'network':
                        if 'pixel_means' in v:
                            v['pixel_means'] = np.array(v['pixel_means'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                config[k] = v
    if config.debug_mode:
        config.train.use_horovod = False
        config.gpus = '0'
