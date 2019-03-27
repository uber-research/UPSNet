# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Modifications Copyright (c) 2019 Uber Technologies, Inc.
# ---------------------------------------------------------------------------
# Based on:
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# and
# --------------------------------------------------------
# mx-maskrcnn
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------



import numpy as np
import numpy.random as npr
from .generate_anchors import generate_anchors
from .anchors import anchors_cython
from upsnet.config.config import config
from upsnet.bbox.bbox_transform import bbox_transform, bbox_overlaps
from upsnet.rpn.generate_anchors import get_field_of_anchors, compute_targets, unmap
from upsnet.bbox.bbox_transform import bbox_overlaps
import pickle

def assign_anchor(feat_shape, gt_boxes, im_info, feat_stride=16,
                  scales=(8, 16, 32), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'bbox_target': of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    'bbox_inside_weight': *todo* mark the assigned anchors
    'bbox_outside_weight': used to normalize the bbox_loss, all weights sums to RPN_POSITIVE_WEIGHT
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    im_info = im_info[0]
    scales = np.array(scales, dtype=np.float32)
    base_anchors = generate_anchors(base_size=feat_stride, ratios=list(ratios), scales=scales)
    num_anchors = base_anchors.shape[0]
    feat_height, feat_width = feat_shape[-2:]

    # 1. generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = num_anchors
    K = shifts.shape[0]
    all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                           (all_anchors[:, 1] >= -allowed_border) &
                           (all_anchors[:, 2] < im_info[1] + allowed_border) &
                           (all_anchors[:, 3] < im_info[0] + allowed_border))[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.train.rpn_clobber_positives:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.train.rpn_negative_overlap] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.train.rpn_positive_overlap] = 1

        if config.train.rpn_clobber_positives:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.train.rpn_negative_overlap] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(config.train.rpn_fg_fraction * config.train.rpn_batch_size)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        # if DEBUG:
        #     disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = config.train.rpn_batch_size - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array(config.train.rpn_bbox_weights)

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, A * feat_height * feat_width))
    bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
    bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))

    label = {'label': labels,
             'bbox_target': bbox_targets,
             'bbox_weight': bbox_weights}
    return label


def assign_pyramid_anchor(gt_boxes, im_info, feat_strides=(64, 32, 16, 8, 4),
                            scales=(8,), ratios=(0.5, 1, 2), allowed_border=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param ratios: aspect ratios of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: tuple
    labels: of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    bbox_targets: of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    bbox_weights: mark the assigned anchors
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    DEBUG = False
    im_info = im_info[0]
    # clean up boxes
    nonneg = np.where(gt_boxes[:, 4] != -1)[0]
    gt_boxes = gt_boxes[nonneg]
    scales = np.array(scales, dtype=np.float32)

    anchors_list = []
    anchors_num_list = []
    inds_inside_list = []
    feat_infos = []
    A_list = []
    for i in range(len(feat_strides)):
        base_anchors = generate_anchors(base_size=feat_strides[i], ratios=list(ratios), scales=scales)
        num_anchors = base_anchors.shape[0]
        # feat_height, feat_width = feat_shape[i][-2:]
        feat_height, feat_width, s = im_info[0], im_info[1], feat_strides[i]
        s = s // 4
        feat_height, feat_width = int(np.ceil(feat_height / 2)) // 2, int(np.ceil(feat_width / 2)) // 2,
        while s > 1:
            feat_height, feat_width = int(np.ceil(feat_height / 2)), int(np.ceil(feat_width / 2))
            s = s // 2
        feat_stride = feat_strides[i]
        feat_infos.append([feat_height, feat_width])

        A = num_anchors
        A_list.append(A)
        K = feat_height * feat_width

        # shift_x = np.arange(0, feat_width) * feat_stride
        # shift_y = np.arange(0, feat_height) * feat_stride
        # shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        # all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = anchors_cython(feat_height, feat_width, feat_stride, base_anchors)
        all_anchors = all_anchors.reshape((K * A, 4))

        total_anchors = int(K * A)
        anchors_num_list.append(total_anchors)
        # only keep anchors inside the image
        inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                               (all_anchors[:, 1] >= -allowed_border) &
                               (all_anchors[:, 2] < im_info[1] + allowed_border) &
                               (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
        if DEBUG:
            print('total_anchors', total_anchors)
            print('inds_inside', len(inds_inside))

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        anchors_list.append(anchors)
        inds_inside_list.append(inds_inside)

    # Concat anchors from each level
    anchors = np.concatenate(anchors_list)
    for i in range(1, len(inds_inside_list)):
        inds_inside_list[i] = inds_inside_list[i] + sum(anchors_num_list[:i])
    inds_inside = np.concatenate(inds_inside_list)
    total_anchors = sum(anchors_num_list)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.train.rpn_clobber_positives:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.train.rpn_negative_overlap] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.train.rpn_positive_overlap] = 1

        if config.train.rpn_clobber_positives:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.train.rpn_negative_overlap] = 0
    else:
        labels[:] = 0

    # subsample positive labels if we have too many
    num_fg = int(config.train.rpn_fg_fraction * config.train.rpn_batch_size)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        if DEBUG:
            disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = config.train.rpn_batch_size - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        if DEBUG:
            disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

    bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_weights[labels == 1, :] = np.array(config.train.rpn_bbox_weights)

    if DEBUG:
        _sums = bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = np.sum(labels == 1)
        means = _sums / (_counts + 1e-14)
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print('means', means)
        print('stdevs', stds)
    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        if gt_boxes.size > 0:
            print('rpn: max max_overlaps', np.max(max_overlaps))
        print('rpn: num_positives', np.sum(labels == 1))
        print('rpn: num_negatives', np.sum(labels == 0))
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        print('rpn: num_positive avg', _fg_sum / _count)
        print('rpn: num_negative avg', _bg_sum / _count)

    # resahpe
    label_list = list()
    bbox_target_list = list()
    bbox_weight_list = list()
    anchors_num_range = [0] + anchors_num_list
    for i in range(len(feat_strides)):
        feat_height, feat_width = feat_infos[i]
        A = A_list[i]
        label = labels[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        bbox_target = bbox_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        bbox_weight = bbox_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]

        label = label.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        label = label.reshape((1, A * feat_height * feat_width))
        bbox_target = bbox_target.reshape((1, feat_height*feat_width, A * 4)).transpose(0, 2, 1)
        bbox_weight = bbox_weight.reshape((1, feat_height*feat_width, A * 4)).transpose((0, 2, 1))

        label_list.append(label)
        bbox_target_list.append(bbox_target)
        bbox_weight_list.append(bbox_weight)

    label_concat = np.concatenate(label_list, axis=1)
    bbox_target_concat = np.concatenate(bbox_target_list, axis=2)
    bbox_weight_concat = np.concatenate(bbox_weight_list, axis=2)

    label= {'label': label_concat,
            'bbox_target': bbox_target_concat,
            'bbox_weight': bbox_weight_concat}
    return label



def add_rpn_blobs(blobs, im_scales, roidb):
    """Add blobs needed training RPN-only and end-to-end Faster R-CNN models."""
    if config.network.has_fpn:
        # RPN applied to many feature levels, as in the FPN paper
        foas = []
        for field_stride in config.network.rpn_feat_stride:
            anchor_sizes = (config.network.anchor_scales[0] * field_stride,)
            anchor_aspect_ratios = config.network.anchor_ratios
            foa = get_field_of_anchors(
                field_stride, anchor_sizes, anchor_aspect_ratios
            )
            foas.append(foa)
        all_anchors = np.concatenate([f.field_of_anchors for f in foas])
    else:
        foa = get_field_of_anchors(
            config.network.rpn_feat_stride, np.array(config.network.anchor_scales) * config.network.rpn_feat_stride, config.network.anchor_ratios
        )
        all_anchors = foa.field_of_anchors

    for im_i, entry in enumerate(roidb):
        scale = im_scales[im_i]
        im_height = np.round(entry['height'] * scale)
        im_width = np.round(entry['width'] * scale)
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0)
        )[0]
        crowd_gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 1)
        )[0]
        gt_rois = entry['boxes'][gt_inds, :] * scale
        crowd_gt_rois = entry['boxes'][crowd_gt_inds, :] * scale
        # TODO(rbg): gt_boxes is poorly named;
        # should be something like 'gt_rois_info'
        gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
        gt_boxes[:, 0] = im_i  # batch inds
        gt_boxes[:, 1:5] = gt_rois
        gt_boxes[:, 5] = entry['gt_classes'][gt_inds]
        im_info = np.array([[im_height, im_width, scale]], dtype=np.float32)
        blobs['im_info'].append(im_info)

        # Add RPN targets
        if config.network.has_fpn:
            # RPN applied to many feature levels, as in the FPN paper
            rpn_blobs = _get_rpn_blobs(
                im_height, im_width, foas, all_anchors, gt_rois)
            for i, lvl in enumerate(config.network.rpn_feat_stride):
                for k, v in rpn_blobs[i].items():
                    blobs[k + '_fpn' + str(lvl)].append(v)
        else:
            # Classical RPN, applied to a single feature level
            rpn_blobs = _get_rpn_blobs(
                im_height, im_width, [foa], all_anchors, gt_rois)
            for k, v in rpn_blobs.items():
                blobs[k].append(v)

    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)

    # valid_keys = [
    #     'has_visible_keypoints', 'boxes', 'segms', 'seg_areas', 'gt_classes',
    #     'gt_overlaps', 'is_crowd', 'box_to_gt_ind_map', 'gt_keypoints'
    # ]
    valid_keys = [
        'boxes', 'segms', 'seg_areas', 'gt_classes',
        'gt_overlaps', 'is_crowd', 'box_to_gt_ind_map'
    ]
    minimal_roidb = [{} for _ in range(len(roidb))]
    for i, e in enumerate(roidb):
        for k in valid_keys:
            if k in e:
                minimal_roidb[i][k] = e[k]
    blobs['roidb'] = minimal_roidb

    # Always return valid=True, since RPN minibatches are valid by design
    return True

def _get_rpn_blobs(im_height, im_width, foas, all_anchors, gt_boxes):
    total_anchors = all_anchors.shape[0]
    straddle_thresh = config.train.rpn_straddle_thresh

    if straddle_thresh >= 0:
        # Only keep anchors inside the image by a margin of straddle_thresh
        # Set TRAIN.RPN_STRADDLE_THRESH to -1 (or a large value) to keep all
        # anchors
        inds_inside = np.where(
            (all_anchors[:, 0] >= -straddle_thresh) &
            (all_anchors[:, 1] >= -straddle_thresh) &
            (all_anchors[:, 2] < im_width + straddle_thresh) &
            (all_anchors[:, 3] < im_height + straddle_thresh)
        )[0]
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
    else:
        inds_inside = np.arange(all_anchors.shape[0])
        anchors = all_anchors
    num_inside = len(inds_inside)

    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_inside,), dtype=np.int32)
    labels.fill(-1)
    if len(gt_boxes) > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        anchor_by_gt_overlap = bbox_overlaps(anchors, gt_boxes)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                                anchor_to_gt_argmax]

        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax,
            np.arange(anchor_by_gt_overlap.shape[1])
        ]
        # Find all anchors that share the max overlap amount
        # (this includes many ties)
        anchors_with_max_overlap = np.where(
            anchor_by_gt_overlap == gt_to_anchor_max
        )[0]

        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        labels[anchors_with_max_overlap] = 1
        # Fg label: above threshold IOU
        labels[anchor_to_gt_max >= config.train.rpn_positive_overlap] = 1


    # subsample positive labels if we have too many
    num_fg = int(config.train.rpn_fg_fraction * config.train.rpn_batch_size)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False
        )
        # print('assign_anchor debug use first 128 fg')
        # labels[fg_inds[-(len(fg_inds) - num_fg):]] = -1
        labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]

    # subsample negative labels if we have too many
    # (samples with replacement, but since the set of bg inds is large most
    # samples will not have repeats)
    num_bg = config.train.rpn_batch_size - np.sum(labels == 1)
    bg_inds = np.where(anchor_to_gt_max < config.train.rpn_negative_overlap)[0]
    if len(bg_inds) > num_bg:
        # enable_inds = bg_inds[np.random.randint(len(bg_inds), size=num_bg)]
        enable_inds = bg_inds[np.random.choice(len(bg_inds), num_bg, replace=False)]
        # print('assign_anchor debug use first 128 bg')
        # labels[bg_inds[:num_bg]] = 0
        labels[enable_inds] = 0
    bg_inds = np.where(labels == 0)[0]

    bbox_targets = np.zeros((num_inside, 4), dtype=np.float32)
    bbox_targets[fg_inds, :] = compute_targets(
        anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :]
    )

    # Bbox regression loss has the form:
    #   loss(x) = weight_outside * L(weight_inside * x)
    # Inside weights allow us to set zero loss on an element-wise basis
    # Bbox regression is only trained on positive examples so we set their
    # weights to 1.0 (or otherwise if config is different) and 0 otherwise
    bbox_inside_weights = np.zeros((num_inside, 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = (1.0, 1.0, 1.0, 1.0)

    # The bbox regression loss only averages by the number of images in the
    # mini-batch, whereas we need to average by the total number of example
    # anchors selected
    # Outside weights are used to scale each element-wise loss so the final
    # average over the mini-batch is correct
    bbox_outside_weights = np.zeros((num_inside, 4), dtype=np.float32)
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    bbox_outside_weights[labels == 1, :] = 1.0 / num_examples
    bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

    # Map up to original set of anchors
    labels = unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = unmap(
        bbox_targets, total_anchors, inds_inside, fill=0
    )
    bbox_inside_weights = unmap(
        bbox_inside_weights, total_anchors, inds_inside, fill=0
    )
    bbox_outside_weights = unmap(
        bbox_outside_weights, total_anchors, inds_inside, fill=0
    )

    # Split the generated labels, etc. into labels per each field of anchors
    blobs_out = []
    start_idx = 0
    for foa in foas:
        H = foa.field_size
        W = foa.field_size
        A = foa.num_cell_anchors
        end_idx = start_idx + H * W * A
        _labels = labels[start_idx:end_idx]
        _bbox_targets = bbox_targets[start_idx:end_idx, :]
        _bbox_inside_weights = bbox_inside_weights[start_idx:end_idx, :]
        _bbox_outside_weights = bbox_outside_weights[start_idx:end_idx, :]
        start_idx = end_idx

        # labels output with shape (1, A, height, width)
        _labels = _labels.reshape((1, H, W, A)).transpose(0, 3, 1, 2)
        # bbox_targets output with shape (1, 4 * A, height, width)
        _bbox_targets = _bbox_targets.reshape(
            (1, H, W, A * 4)).transpose(0, 3, 1, 2)
        # bbox_inside_weights output with shape (1, 4 * A, height, width)
        _bbox_inside_weights = _bbox_inside_weights.reshape(
            (1, H, W, A * 4)).transpose(0, 3, 1, 2)
        # bbox_outside_weights output with shape (1, 4 * A, height, width)
        _bbox_outside_weights = _bbox_outside_weights.reshape(
            (1, H, W, A * 4)).transpose(0, 3, 1, 2)
        blobs_out.append(
            dict(
                rpn_labels_int32_wide=_labels,
                rpn_bbox_targets_wide=_bbox_targets,
                rpn_bbox_inside_weights_wide=_bbox_inside_weights,
                rpn_bbox_outside_weights_wide=_bbox_outside_weights
            )
        )
    return blobs_out[0] if len(blobs_out) == 1 else blobs_out
