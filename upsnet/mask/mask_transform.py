# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Modifications Copyright (c) 2019 Uber Technologies, Inc.
# ---------------------------------------------------------------------------
# Based on:
# ---------------------------------------------------------------------------
# Detectron
# Copyright (c) 2017-present, Facebook, Inc.
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
# ---------------------------------------------------------------------------

import math
import numpy as np
import cv2
import pycocotools.mask as mask_util
from upsnet.config.config import config
from upsnet.bbox.bbox_transform import bbox_overlaps

def get_gt_masks(gt_masks, size):
    num_mask = gt_masks.shape[0]
    processed_masks = np.zeros((num_mask, size[0], size[1]))
    for i in range(num_mask):
        processed_masks[i,:,:] = cv2.resize(gt_masks[i].astype('float32'), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return processed_masks



def mask_overlap(box1, box2, mask1, mask2):
    """
    This function calculate region IOU when masks are
    inside different boxes
    Returns:
        intersection over unions of this two masks
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 > x2 or y1 > y2:
        return 0
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    # get masks in the intersection part
    start_ya = y1 - box1[1]
    start_xa = x1 - box1[0]
    inter_maska = mask1[start_ya: start_ya + h, start_xa:start_xa + w]

    start_yb = y1 - box2[1]
    start_xb = x1 - box2[0]
    inter_maskb = mask2[start_yb: start_yb + h, start_xb:start_xb + w]

    assert inter_maska.shape == inter_maskb.shape

    inter = np.logical_and(inter_maskb, inter_maska).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)


def intersect_box_mask(ex_box, gt_box, gt_mask):
    """
    This function calculate the intersection part of a external box
    and gt_box, mask it according to gt_mask
    Args:
        ex_box: external ROIS
        gt_box: ground truth boxes
        gt_mask: ground truth masks, not been resized yet
    Returns:
        regression_target: logical numpy array
    """
    x1 = max(ex_box[0], gt_box[0])
    y1 = max(ex_box[1], gt_box[1])
    x2 = min(ex_box[2], gt_box[2])
    y2 = min(ex_box[3], gt_box[3])
    if x1 > x2 or y1 > y2:
        return np.zeros((21, 21), dtype=bool)
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    ex_starty = y1 - ex_box[1]
    ex_startx = x1 - ex_box[0]

    inter_maskb = gt_mask[y1:y2+1 , x1:x2+1]
    regression_target = np.zeros((ex_box[3] - ex_box[1] + 1, ex_box[2] - ex_box[0] + 1))
    regression_target[ex_starty: ex_starty + h, ex_startx: ex_startx + w] = inter_maskb

    return regression_target


def mask_aggregation(boxes, masks, mask_weights, im_width, im_height, binary_thresh=0.4):
    """
    This function implements mask voting mechanism to give finer mask
    n is the candidate boxes (masks) number
    Args:
        masks: All masks need to be aggregated (n x sz x sz)
        mask_weights: class score associated with each mask (n x 1)
        boxes: tight box enclose each mask (n x 4)
        im_width, im_height: image information
    """
    assert boxes.shape[0] == len(masks) and boxes.shape[0] == mask_weights.shape[0]
    im_mask = np.zeros((im_height, im_width))
    for mask_ind in range(len(masks)):
        box = np.round(boxes[mask_ind]).astype(int)
        mask = (masks[mask_ind] >= binary_thresh).astype(float)

        mask_weight = mask_weights[mask_ind]
        im_mask[box[1]:box[3]+1, box[0]:box[2]+1] += mask * mask_weight
    [r, c] = np.where(im_mask >= binary_thresh)
    if len(r) == 0 or len(c) == 0:
        min_y = np.ceil(im_height / 2).astype(int)
        min_x = np.ceil(im_width / 2).astype(int)
        max_y = min_y
        max_x = min_x
    else:
        min_y = np.min(r)
        min_x = np.min(c)
        max_y = np.max(r)
        max_x = np.max(c)

    clipped_mask = im_mask[min_y:max_y+1, min_x:max_x+1]
    clipped_box = np.array((min_x, min_y, max_x, max_y), dtype=np.float32)
    return clipped_mask, clipped_box


def flip_segms(segms, height, width):
    """Left/right flip each mask in a list of masks."""
    def _flip_poly(poly, width):
        flipped_poly = np.array(poly)
        flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
        return flipped_poly.tolist()

    def _flip_rle(rle, height, width):
        if 'counts' in rle and type(rle['counts']) == list:
            # Magic RLE format handling painfully discovered by looking at the
            # COCO API showAnns function.
            rle = mask_util.frPyObjects([rle], height, width)
        mask = mask_util.decode(rle)
        mask = mask[:, ::-1, :]
        rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
        return rle

    flipped_segms = []
    for segm in segms:
        if type(segm) == list:
            # Polygon format
            flipped_segms.append([_flip_poly(poly, width) for poly in segm])
        else:
            # RLE format
            assert type(segm) == dict
            flipped_segms.append(_flip_rle(segm, height, width))
    return flipped_segms


def polys_to_mask(polygons, height, width):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed inside a height x width image. The resulting
    mask is therefore of shape (height, width).
    """
    rle = mask_util.frPyObjects(polygons, height, width)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


def mask_to_bbox(mask):
    """Compute the tight bounding box of a binary mask."""
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]
    return np.array((x0, y0, x1, y1), dtype=np.float32)


def polys_to_mask_wrt_box(polygons, box, M):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed in the given box and rasterized to an M x M
    mask. The resulting mask is therefore of shape (M, M).
    """
    w = box[2] - box[0]
    h = box[3] - box[1]

    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    polygons_norm = []
    for poly in polygons:
        p = np.array(poly, dtype=np.float32)
        p[0::2] = (p[0::2] - box[0]) * M / w
        p[1::2] = (p[1::2] - box[1]) * M / h
        polygons_norm.append(p)

    rle = mask_util.frPyObjects(polygons_norm, M, M)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys

def add_mask_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    """Add Mask R-CNN specific blobs to the input blob dictionary."""
    # Prepare the mask targets by associating one gt mask to each training roi
    # that has a fg (non-bg) class label.
    M = config.network.mask_size
    polys_gt_inds = np.where(
        (roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0)
    )[0]
    polys_gt = [roidb['segms'][i] for i in polys_gt_inds]
    boxes_from_polys = polys_to_boxes(polys_gt)
    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_mask = blobs['labels_int32'].copy()
    roi_has_mask[roi_has_mask > 0] = 1

    if fg_inds.shape[0] > 0:
        # Class labels for the foreground rois
        mask_class_labels = blobs['labels_int32'][fg_inds]
        masks = np.zeros((fg_inds.shape[0], M**2), dtype=np.int32)

        # Find overlap between all foreground rois and the bounding boxes
        # enclosing each segmentation
        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False)
        )
        # Map from each fg rois to the index of the mask with highest overlap
        # (measured by bbox overlap)
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # add fg targets
        for i in range(rois_fg.shape[0]):
            fg_polys_ind = fg_polys_inds[i]
            poly_gt = polys_gt[fg_polys_ind]
            roi_fg = rois_fg[i]
            # Rasterize the portion of the polygon mask within the given fg roi
            # to an M x M binary image
            mask = polys_to_mask_wrt_box(poly_gt, roi_fg, M)
            mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
            masks[i, :] = np.reshape(mask, M**2)
    else:  # If there are no fg masks (it does happen)
        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        # rois_fg is actually one background roi, but that's ok because ...
        rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        # We give it an -1's blob (ignore label)
        masks = -np.ones((1, M**2), dtype=np.int32)
        # We label it with class = 0 (background)
        mask_class_labels = np.zeros((1, ), dtype=np.float32)
        # Mark that the first roi has a mask
        roi_has_mask[0] = 1

    masks = _expand_to_class_specific_mask_targets(masks, mask_class_labels)

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * np.ones((rois_fg.shape[0], 1), dtype=np.float32)
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))

    # Update blobs dict with Mask R-CNN blobs
    blobs['mask_rois'] = rois_fg
    blobs['roi_has_mask_int32'] = roi_has_mask
    blobs['mask_int32'] = masks


def _expand_to_class_specific_mask_targets(masks, mask_class_labels):
    """Expand masks from shape (#masks, M ** 2) to (#masks, #classes * M ** 2)
    to encode class specific mask targets.
    """
    assert masks.shape[0] == mask_class_labels.shape[0]
    M = config.network.mask_size

    # Target values of -1 are "don't care" / ignore labels
    mask_targets = -np.ones(
        (masks.shape[0], config.dataset.num_classes * M**2), dtype=np.int32
    )

    for i in range(masks.shape[0]):
        cls = int(mask_class_labels[i])
        start = M**2 * cls
        end = start + M**2
        # Ignore background instance
        # (only happens when there is no fg samples in an image)
        if cls > 0:
            mask_targets[i, start:end] = masks[i, :]

    return mask_targets
