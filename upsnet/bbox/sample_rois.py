# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Modifications Copyright (c) 2019 Uber Technologies, Inc.
# --------------------------------------------------------
# Based on:
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import numpy.random as npr
import cv2
from upsnet.config.config import config
from upsnet.bbox.bbox_transform import bbox_overlaps, bbox_transform, bbox_transform_inv
from upsnet.bbox.bbox_regression import expand_bbox_regression_targets
from upsnet.mask.mask_transform import intersect_box_mask, add_mask_rcnn_blobs
import time

def compute_assign_targets(rois, threshold=[[np.inf, 448], [448, 224], [224, 112], [112, 0]]):
    rois_area = np.sqrt((rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1))
    num_rois = np.shape(rois)[0]
    assign_levels = np.zeros(num_rois, dtype=np.uint8)
    for i, stride in enumerate(config.network.rcnn_feat_stride):
        thd = threshold[i]
        idx = np.logical_and(thd[1] <= rois_area, rois_area < thd[0])
        assign_levels[idx] = stride

    assert 0 not in assign_levels, "All rois should assign to specify levels."
    return assign_levels


def compute_mask_and_label(rois, labels, gt_masks, gt_assignment, num_classes, mask_size):
    n_rois = rois.shape[0]
    mask_targets = np.zeros((n_rois, num_classes, mask_size, mask_size))
    mask_weights = np.zeros((n_rois, num_classes, 1, 1))
    for n in range(n_rois):
        target = gt_masks[gt_assignment[n], int(rois[n, 2]): int(rois[n, 4]), int(rois[n, 1]): int(rois[n, 3])]
        if labels[n] == 0 or target.size == 0:
            continue
        mask = np.zeros(target.shape)
        mask[target == 1] = 1
        mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
        mask_targets[n, int(labels[n]), ...] = mask
        mask_weights[n, int(labels[n]), ...] = 1
    return mask_targets, mask_weights


def sample_rois(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    rois_per_image = int(config.train.batch_rois)
    fg_rois_per_image = int(np.round(config.train.fg_fraction * rois_per_image))
    max_overlaps = roidb['max_overlaps']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= config.train.fg_thresh)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        # print('sample rois use first 128 fg')
        # fg_inds = fg_inds[:fg_rois_per_this_image]
        fg_inds = npr.choice(
            fg_inds, size=fg_rois_per_this_image, replace=False
        )

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(
        (max_overlaps < config.train.bg_thresh_hi) &
        (max_overlaps >= config.train.bg_thresh_lo)
    )[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        # print('sample rois use first 384 bg')
        # bg_inds = bg_inds[:bg_rois_per_this_image]
        bg_inds = npr.choice(
            bg_inds, size=bg_rois_per_this_image, replace=False
        )

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # print(keep_inds.shape)
    # Label is the class each RoI has max overlap with
    sampled_labels = roidb['max_classes'][keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes = roidb['boxes'][keep_inds]

    if 'bbox_targets' not in roidb:
        gt_inds = np.where(roidb['gt_classes'] > 0)[0]
        gt_boxes = roidb['boxes'][gt_inds, :]
        gt_assignments = gt_inds[roidb['box_to_gt_ind_map'][keep_inds]]
        bbox_targets = _compute_targets(
            sampled_boxes, gt_boxes[gt_assignments, :], sampled_labels
        )
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(bbox_targets)
    else:
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(
            roidb['bbox_targets'][keep_inds, :]
        )

    bbox_outside_weights = np.array(
        bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype
    )

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * np.ones((sampled_rois.shape[0], 1), np.float32)
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    nongt_inds = np.where(roidb['gt_classes'][keep_inds] == 0)[0]

    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_int32=sampled_labels.astype(np.int32, copy=False),
        rois=sampled_rois,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights,
        nongt_inds=nongt_inds
    )

    # Optionally add Mask R-CNN blobs
    if config.network.has_mask_head:
        add_mask_rcnn_blobs(
            blob_dict, sampled_boxes, roidb, im_scale, batch_idx
        )

    return blob_dict

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform_inv(
        ex_rois, gt_rois, config.network.bbox_reg_weights
    )
    return np.hstack((labels[:, np.newaxis], targets)).astype(
        np.float32, copy=False
    )


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.
    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = 2 if config.network.cls_agnostic_bbox_reg else config.dataset.num_classes

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_bbox_reg_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights
