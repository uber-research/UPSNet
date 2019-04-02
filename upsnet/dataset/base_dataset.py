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
# Written by Yuwen Xiong, Hengshuang Zhao
# ---------------------------------------------------------------------------

from __future__ import print_function

import os
import sys
import torch
import torch.utils.data

import pickle, gzip
import numpy as np
import scipy.io as sio
import cv2
import json
import torch.multiprocessing as multiprocessing
import time
from PIL import Image, ImageDraw
from collections import defaultdict, Sequence
from pycocotools.cocoeval import COCOeval

from upsnet.config.config import config
from upsnet.rpn.assign_anchor import add_rpn_blobs
from upsnet.bbox import bbox_transform
from upsnet.bbox.sample_rois import sample_rois
import networkx as nx
from lib.utils.logging import logger

import pycocotools.mask as mask_util

# panoptic visualization
vis_panoptic = False

class PQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'iou': iou, 'tp':tp, 'fp':fp, 'fn':fn}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self):

        self.flip = None
        self.roidb = None
        self.phase = None
        self.num_classes = None
        self.result_path = None

    def __len__(self):
        return len(self.roidb)

    def get_image_blob(self, roidb):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = len(roidb)
        # Sample random scales to use for each image in this batch
        scale_inds = np.random.randint(
            0, high=len(config.train.scales), size=num_images
        )
        processed_ims = []
        im_scales = []
        for i in range(num_images):
            im = cv2.imread(roidb[i]['image'])
            assert im is not None, \
                'Failed to read image \'{}\''.format(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            target_size = config.train.scales[scale_inds[i]]
            im, im_scale = self.prep_im_for_blob(
                im, config.network.pixel_means, [target_size], config.train.max_size
            )
            im_scales.append(im_scale[0])
            processed_ims.append(im[0].transpose(2, 0, 1))

        # Create a blob to hold the input images
        assert len(processed_ims) == 1
        blob = processed_ims[0]

        return blob, im_scales

    def prep_im_for_blob(self, im, pixel_means, target_sizes, max_size):
        """Prepare an image for use as a network input blob. Specially:
          - Subtract per-channel pixel mean
          - Convert to float32
          - Rescale to each of the specified target size (capped at max_size)
        Returns a list of transformed images, one for each target size. Also returns
        the scale factors that were used to compute each returned image.
        """
        im = im.astype(np.float32, copy=False)
        if config.network.use_caffe_model:
            im -= pixel_means.reshape((1, 1, -1))
        else:
            im /= 255.0
            im -= np.array([[[0.485, 0.456, 0.406]]])
            im /= np.array([[[0.229, 0.224, 0.225]]])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        ims = []
        im_scales = []
        for target_size in target_sizes:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            ims.append(im)
            im_scales.append(im_scale)
        return ims, im_scales


    def evaluate_all(self, all_boxes, all_segms, output_dir):
        all_results = self.evaluate_boxes(all_boxes, output_dir)
        self.evaluate_masks(all_boxes, all_segms, output_dir)
        return all_results

    def evaluate_boxes(self, all_boxes, output_dir):
        res_file = os.path.join(
            output_dir, 'bbox_' + self.dataset.name + '_results.json'
        )
        results = []
        for cls_ind, cls in enumerate(self.dataset.classes):
            if cls == '__background__':
                continue
            if cls_ind >= len(all_boxes):
                break
            cat_id = self.dataset.category_to_id_map[cls]
            results.extend(self.bbox_results_one_category(all_boxes[cls_ind], cat_id))
        if logger:
            logger.info('Writing bbox results json to: {}'.format(os.path.abspath(res_file)))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

        coco_dt = self.dataset.COCO.loadRes(str(res_file))
        coco_eval = COCOeval(self.dataset.COCO, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self.log_detection_eval_metrics(coco_eval, os.path.join(output_dir, 'detection_results.txt'))

        return coco_eval.stats

    def evaluate_masks(self, all_boxes, all_segms, output_dir):
        pass

    def evaluate_panoptic(self, pred_pans_2ch, output_dir):

        sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'lib', 'dataset_devkit'))

        from panopticapi.utils import IdGenerator

        def get_gt(pan_gt_json_file=None, pan_gt_folder=None):
            if pan_gt_json_file is None:
                pan_gt_json_file = self.panoptic_json_file
            if pan_gt_folder is None:
                pan_gt_folder = self.panoptic_gt_folder
            with open(pan_gt_json_file, 'r') as f:
                pan_gt_json = json.load(f)
            files = [item['file_name'] for item in pan_gt_json['images']]
            cpu_num = multiprocessing.cpu_count()
            files_split = np.array_split(files, cpu_num)
            workers = multiprocessing.Pool(processes=cpu_num)
            processes = []
            for proc_id, files_set in enumerate(files_split):
                p = workers.apply_async(BaseDataset._load_image_single_core, (proc_id, files_set, pan_gt_folder))
                processes.append(p)
            workers.close()
            workers.join()
            pan_gt_all = []
            for p in processes:
                pan_gt_all.extend(p.get())

            categories = pan_gt_json['categories']
            categories = {el['id']: el for el in categories}
            color_gererator = IdGenerator(categories)

            return pan_gt_all, pan_gt_json, categories, color_gererator

        def get_pred(pan_2ch_all, color_gererator, cpu_num=None):
            if cpu_num is None:
                cpu_num = multiprocessing.cpu_count()
            pan_2ch_split = np.array_split(pan_2ch_all, cpu_num)
            workers = multiprocessing.Pool(processes=cpu_num)
            processes = []
            for proc_id, pan_2ch_set in enumerate(pan_2ch_split):
                p = workers.apply_async(BaseDataset._converter_2ch_single_core, (proc_id, pan_2ch_set, color_gererator))
                processes.append(p)
            workers.close()
            workers.join()
            annotations, pan_all = [], []
            for p in processes:
                p = p.get()
                annotations.extend(p[0])
                pan_all.extend(p[1])
            pan_json = {'annotations': annotations}
            return pan_all, pan_json

        def save_image(images, save_folder, gt_json, colors=None):
            os.makedirs(save_folder, exist_ok=True)
            names = [os.path.join(save_folder, item['file_name'].replace('_leftImg8bit', '').replace('jpg', 'png').replace('jpeg', 'png')) for item in gt_json['images']]
            cpu_num = multiprocessing.cpu_count()
            images_split = np.array_split(images, cpu_num)
            names_split = np.array_split(names, cpu_num)
            workers = multiprocessing.Pool(processes=cpu_num)
            for proc_id, (images_set, names_set) in enumerate(zip(images_split, names_split)):
                workers.apply_async(BaseDataset._save_image_single_core, (proc_id, images_set, names_set, colors))
            workers.close()
            workers.join()

        def pq_compute(gt_jsons, pred_jsons, gt_pans, pred_pans, categories):
            start_time = time.time()
            # from json and from numpy
            gt_image_jsons = gt_jsons['images']
            gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
            cpu_num = multiprocessing.cpu_count()
            gt_jsons_split, pred_jsons_split = np.array_split(gt_jsons, cpu_num), np.array_split(pred_jsons, cpu_num)
            gt_pans_split, pred_pans_split = np.array_split(gt_pans, cpu_num), np.array_split(pred_pans, cpu_num)
            gt_image_jsons_split = np.array_split(gt_image_jsons, cpu_num)

            workers = multiprocessing.Pool(processes=cpu_num)
            processes = []
            for proc_id, (gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set) in enumerate(zip(gt_jsons_split, pred_jsons_split, gt_pans_split, pred_pans_split, gt_image_jsons_split)):
                p = workers.apply_async(BaseDataset._pq_compute_single_core, (proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories))
                processes.append(p)
            workers.close()
            workers.join()
            pq_stat = PQStat()
            for p in processes:
                pq_stat += p.get()
            metrics = [("All", None), ("Things", True), ("Stuff", False)]
            results = {}
            for name, isthing in metrics:
                results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
                if name == 'All':
                    results['per_class'] = per_class_results

            if logger:
                logger.info("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
                logger.info("-" * (10 + 7 * 4))
                for name, _isthing in metrics:
                    logger.info("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(name, 100 * results[name]['pq'], 100 * results[name]['sq'], 100 * results[name]['rq'], results[name]['n']))

                logger.info("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s}".format("IDX", "PQ", "SQ", "RQ", "IoU", "TP", "FP", "FN"))
                for idx, result in results['per_class'].items():
                    logger.info("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d}".format(idx, 100 * result['pq'], 100 * result['sq'], 100 * result['rq'], result['iou'], result['tp'],
                                                                                             result['fp'], result['fn']))

            t_delta = time.time() - start_time
            print("Time elapsed: {:0.2f} seconds".format(t_delta))
            return results


        # if eval for test-dev, since there is no gt we simply retrieve image names from image_info json files
        # with open(self.panoptic_json_file, 'r') as f:
        #     gt_json = json.load(f)
        #     gt_json['images'] = sorted(gt_json['images'], key=lambda x: x['id'])
        # other wise:
        gt_pans, gt_json, categories, color_gererator = get_gt()
        
        pred_pans, pred_json = get_pred(pred_pans_2ch, color_gererator)
        save_image(pred_pans_2ch, os.path.join(output_dir, 'pan_2ch'), gt_json)
        save_image(pred_pans, os.path.join(output_dir, 'pan'), gt_json)
        json.dump(gt_json, open(os.path.join(output_dir, 'gt.json'), 'w'))
        json.dump(pred_json, open(os.path.join(output_dir, 'pred.json'), 'w'))
        results = pq_compute(gt_json, pred_json, gt_pans, pred_pans, categories)

        return results

    def get_unified_pan_result(self, segs, pans, cls_inds, stuff_area_limit=4 * 64 * 64):
        pred_pans_2ch = []

        for (seg, pan, cls_ind) in zip(segs, pans, cls_inds):
            pan_seg = pan.copy()
            pan_ins = pan.copy()
            id_last_stuff = config.dataset.num_seg_classes - config.dataset.num_classes
            ids = np.unique(pan)
            ids_ins = ids[ids > id_last_stuff]
            pan_ins[pan_ins <= id_last_stuff] = 0
            for idx, id in enumerate(ids_ins):
                region = (pan_ins == id)
                if id == 255:
                    pan_seg[region] = 255
                    pan_ins[region] = 0
                    continue
                cls, cnt = np.unique(seg[region], return_counts=True)
                if cls[np.argmax(cnt)] == cls_ind[id - id_last_stuff - 1] + id_last_stuff:
                    pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                    pan_ins[region] = idx + 1
                else:
                    if np.max(cnt) / np.sum(cnt) >= 0.5 and cls[np.argmax(cnt)] <= id_last_stuff:
                        pan_seg[region] = cls[np.argmax(cnt)]
                        pan_ins[region] = 0 
                    else:
                        pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                        pan_ins[region] = idx + 1

            idx_sem = np.unique(pan_seg)
            for i in range(idx_sem.shape[0]):
                if idx_sem[i] <= id_last_stuff:
                    area = pan_seg == idx_sem[i]
                    if (area).sum() < stuff_area_limit:
                        pan_seg[area] = 255

            pan_2ch = np.zeros((pan.shape[0], pan.shape[1], 3), dtype=np.uint8)
            pan_2ch[:, :, 0] = pan_seg
            pan_2ch[:, :, 1] = pan_ins
            pred_pans_2ch.append(pan_2ch)
        return pred_pans_2ch

    def get_combined_pan_result(self, segs, boxes, masks, score_threshold=0.6, fraction_threshold=0.7, stuff_area_limit=4*64*64):
        # suppose ins masks are already sorted in descending order by scores
        boxes_all, masks_all, cls_idxs_all = [], [], []
        boxes_all = []
        import itertools
        import time
        for i in range(len(segs)):
            boxes_i = np.vstack([boxes[j][i] for j in range(1, len(boxes))])
            masks_i = np.array(list(itertools.chain(*[masks[j][i] for j in range(1, len(masks))])))
            cls_idxs_i = np.hstack([np.array([j for _ in boxes[j][i]]).astype(np.int32) for j in range(1, len(boxes))])
            sorted_idxs = np.argsort(boxes_i[:, 4])[::-1]
            boxes_all.append(boxes_i[sorted_idxs])
            masks_all.append(masks_i[sorted_idxs])
            cls_idxs_all.append(cls_idxs_i[sorted_idxs])

        cpu_num = multiprocessing.cpu_count()
        boxes_split = np.array_split(boxes_all, cpu_num)
        cls_idxs_split = np.array_split(cls_idxs_all, cpu_num)
        masks_split = np.array_split(masks_all, cpu_num)
        segs_split = np.array_split(segs, cpu_num)
        workers = multiprocessing.Pool(processes=cpu_num)
        processes = []
        for proc_id, (boxes_set, cls_idxs_set, masks_set, sems_set) in enumerate(zip(boxes_split, cls_idxs_split, masks_split, segs_split)):
            p = workers.apply_async(BaseDataset._merge_pred_single_core, (proc_id, boxes_set, cls_idxs_set, masks_set, sems_set, score_threshold, fraction_threshold, stuff_area_limit))
            processes.append(p)
        workers.close()
        workers.join()
        pan_2ch_all = []
        for p in processes:
            pan_2ch_all.extend(p.get())
        return pan_2ch_all

    @staticmethod
    def _merge_pred_single_core(proc_id, boxes_set, cls_idxs_set, masks_set, sems_set, score_threshold, fraction_threshold, stuff_area_limit):
        from pycocotools.mask import decode as mask_decode
        pan_2ch_all = []
        id_last_stuff = config.dataset.num_seg_classes - config.dataset.num_classes

        for idx_outer in range(len(boxes_set)):
            boxes, scores, cls_idxs, masks = boxes_set[idx_outer][:, :4], boxes_set[idx_outer][:, 4], cls_idxs_set[idx_outer], masks_set[idx_outer]
            sem = sems_set[idx_outer]
            h, w = sem.shape
            ins_mask = np.zeros((h, w), dtype=np.uint8)
            ins_sem = np.zeros((h, w), dtype=np.uint8)
            idx_ins_array = np.zeros(config.dataset.num_classes - 1, dtype=np.uint32)
            for idx_inner in range(len(scores)):
                score, cls_idx, mask = scores[idx_inner], cls_idxs[idx_inner], masks[idx_inner]
                if score < score_threshold:
                    continue
                mask = mask_decode(masks[idx_inner])
                ins_remain = (mask == 1) & (ins_mask == 0)
                if (mask.astype(np.float32).sum() == 0) or (ins_remain.astype(np.float32).sum() / mask.astype(np.float32).sum() < fraction_threshold):
                    continue
                idx_ins_array[cls_idx - 1] += 1
                ins_mask[ins_remain] = idx_ins_array[cls_idx - 1]
                ins_sem[ins_remain] = cls_idx

            idx_sem = np.unique(sem)
            for i in range(idx_sem.shape[0]):
                if idx_sem[i] <= id_last_stuff:
                    area = sem == idx_sem[i]
                    if (area).sum() < stuff_area_limit:
                        sem[area] = 255

            # merge sem and ins, leave conflict region as 255
            pan_2ch = np.zeros((h, w, 3), dtype=np.uint8)
            pan_2ch_c0 = sem.copy()
            pan_2ch_c1 = ins_mask.copy()
            conflict = (sem > id_last_stuff) & (ins_mask == 0)  # sem treat as thing while ins treat as stuff
            pan_2ch_c0[conflict] = 255
            insistence = (ins_mask != 0)  # change sem region to ins thing region
            pan_2ch_c0[insistence] = ins_sem[insistence] + id_last_stuff
            pan_2ch[:, :, 0] = pan_2ch_c0
            pan_2ch[:, :, 1] = pan_2ch_c1
            pan_2ch_all.append(pan_2ch)

        return pan_2ch_all

    @staticmethod
    def _load_image_single_core(proc_id, files_set, folder):
        images = []
        for working_idx, file in enumerate(files_set):
            try:
                image = np.array(Image.open(os.path.join(folder, file)))
                images.append(image)
            except Exception:
                pass
        return images

    @staticmethod
    def _converter_2ch_single_core(proc_id, pan_2ch_set, color_gererator):
        sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'lib', 'dataset_devkit'))
        from panopticapi.utils import rgb2id
        OFFSET = 1000
        VOID = 255
        annotations, pan_all = [], []
        for idx in range(len(pan_2ch_set)):
            pan_2ch = np.uint32(pan_2ch_set[idx])
            pan = OFFSET * pan_2ch[:, :, 0] + pan_2ch[:, :, 1]
            pan_format = np.zeros((pan_2ch.shape[0], pan_2ch.shape[1], 3), dtype=np.uint8)

            l = np.unique(pan)
            segm_info = []
            for el in l:
                sem = el // OFFSET
                if sem == VOID:
                    continue
                mask = pan == el
                if vis_panoptic:
                    color = color_gererator.categories[sem]['color']
                else:
                    color = color_gererator.get_color(sem)
                pan_format[mask] = color
                index = np.where(mask)
                x = index[1].min()
                y = index[0].min()
                width = index[1].max() - x
                height = index[0].max() - y
                segm_info.append({"category_id": sem.item(), "iscrowd": 0, "id": int(rgb2id(color)), "bbox": [x.item(), y.item(), width.item(), height.item()], "area": mask.sum().item()})
            annotations.append({"segments_info": segm_info})
            if vis_panoptic:
                pan_format = Image.fromarray(pan_format)
                draw = ImageDraw.Draw(pan_format)
                for el in l:
                    sem = el // OFFSET
                    if sem == VOID:
                        continue
                    if color_gererator.categories[sem]['isthing'] and el % OFFSET != 0:
                        mask = ((pan == el) * 255).astype(np.uint8)
                        _, contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for c in contour:
                            c = c.reshape(-1).tolist()
                            if len(c) < 4:
                                print('warning: invalid contour')
                                continue
                            draw.line(c, fill='white', width=2)
                pan_format = np.array(pan_format)
            pan_all.append(pan_format)
        return annotations, pan_all

    @staticmethod
    def _pq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories):
        OFFSET = 256 * 256 * 256
        VOID = 0
        pq_stat = PQStat()
        for idx, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(zip(gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set)):
            # if idx % 100 == 0:
            #     logger.info('Compute pq -> Core: {}, {} from {} images processed'.format(proc_id, idx, len(gt_jsons_set)))
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256

            gt_segms = {el['id']: el for el in gt_json['segments_info']}
            pred_segms = {el['id']: el for el in pred_json['segments_info']}

            # predicted segments area calculation + prediction sanity checks
            pred_labels_set = set(el['id'] for el in pred_json['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in categories:
                    raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

            # confusion matrix calculation
            pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
            gt_pred_map = {}
            labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

            # count all matched pairs
            gt_matched = set()
            pred_matched = set()
            tp = 0
            fp = 0
            fn = 0

            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                if gt_label not in gt_segms:
                    continue
                if pred_label not in pred_segms:
                    continue
                if gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                    continue

                union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get(
                    (VOID, pred_label), 0)
                iou = intersection / union
                if iou > 0.5:
                    pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                    pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    tp += 1

            # count false positives
            crowd_labels_dict = {}
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                pq_stat[gt_info['category_id']].fn += 1
                fn += 1

            # count false positives
            for pred_label, pred_info in pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
                # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                pq_stat[pred_info['category_id']].fp += 1
                fp += 1
        # logger.info('Compute pq -> Core: {}, all {} images processed'.format(proc_id, len(gt_jsons_set)))
        return pq_stat

    @staticmethod
    def _save_image_single_core(proc_id, images_set, names_set, colors=None):
        def colorize(gray, palette):
            # gray: numpy array of the label and 1*3N size list palette
            color = Image.fromarray(gray.astype(np.uint8)).convert('P')
            color.putpalette(palette)
            return color

        for working_idx, (image, name) in enumerate(zip(images_set, names_set)):
            if colors is not None:
                image = colorize(image, colors)
            else:
                image = Image.fromarray(image)
            os.makedirs(os.path.dirname(name), exist_ok=True)
            image.save(name)

    def bbox_results_one_category(self, boxes, cat_id):
        results = []
        image_ids = self.dataset.COCO.getImgIds()
        image_ids.sort()
        assert len(boxes) == len(image_ids)
        for i, image_id in enumerate(image_ids):
            dets = boxes[i]
            if isinstance(dets, list) and len(dets) == 0:
                continue
            dets = dets.astype(np.float)
            scores = dets[:, -1]
            xywh_dets = bbox_transform.xyxy_to_xywh(dets[:, 0:4])
            xs = xywh_dets[:, 0]
            ys = xywh_dets[:, 1]
            ws = xywh_dets[:, 2]
            hs = xywh_dets[:, 3]
            results.extend(
                [{'image_id': image_id,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def segms_results_one_category(self, boxes, segms, cat_id):
        results = []
        image_ids = self.dataset.COCO.getImgIds()
        image_ids.sort()
        assert len(boxes) == len(image_ids)
        assert len(segms) == len(image_ids)
        for i, image_id in enumerate(image_ids):
            dets = boxes[i]
            rles = segms[i]

            if isinstance(dets, list) and len(dets) == 0:
                continue

            dets = dets.astype(np.float)
            scores = dets[:, -1]

            results.extend(
                [{'image_id': image_id,
                  'category_id': cat_id,
                  'segmentation': rles[k],
                  'score': scores[k]}
                 for k in range(dets.shape[0])])

        return results

    def log_detection_eval_metrics(self, coco_eval, log_file):

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        class tee:
            def __init__(self, *files):
                self.files = files

            def write(self, obj):
                for f in self.files:
                    f.write(obj)

        stdout = sys.stdout
        sys.stdout = tee(sys.stdout, open(log_file, 'w'))

        IoU_lo_thresh = 0.5
        for IoU_hi_thresh in [0.95, 0.5]:
            ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
            ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
            # precision has dims (iou, recall, cls, area range, max dets)
            # area range index 0: all area ranges
            # max dets index 2: 100 per image
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
            ap_default = np.mean(precision[precision > -1])
            if logger:
                logger.info(
                    '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
                        IoU_lo_thresh, IoU_hi_thresh))
            for cls_ind, cls in enumerate(self.dataset.classes):
                if cls == '__background__':
                    continue
                # minus 1 because of __background__
                precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
                ap = np.mean(precision[precision > -1])
                if logger:
                    logger.info('{:.3f}'.format(ap))
            if logger:
                logger.info('{:.3f}'.format(ap_default))
        if logger:
            logger.info('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

        sys.stdout = stdout

    def evaluate_box_proposals(
            self, roidb, thresholds=None, area='all', limit=None
    ):
        """Evaluate detection proposal recall metrics. This function is a much
        faster alternative to the official COCO API recall evaluation code. However,
        it produces slightly different results.
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {
            'all': 0,
            'small': 1,
            'medium': 2,
            'large': 3,
            '96-128': 4,
            '128-256': 5,
            '256-512': 6,
            '512-inf': 7}
        area_ranges = [
            [0 ** 2, 1e5 ** 2],  # all
            [0 ** 2, 32 ** 2],  # small
            [32 ** 2, 96 ** 2],  # medium
            [96 ** 2, 1e5 ** 2],  # large
            [96 ** 2, 128 ** 2],  # 96-128
            [128 ** 2, 256 ** 2],  # 128-256
            [256 ** 2, 512 ** 2],  # 256-512
            [512 ** 2, 1e5 ** 2]]  # 512-inf
        assert area in areas, 'Unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for entry in roidb:
            gt_inds = np.where(
                (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_areas = entry['seg_areas'][gt_inds]
            valid_gt_inds = np.where(
                (gt_areas >= area_range[0]) & (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)
            non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
            boxes = entry['boxes'][non_gt_inds, :]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]
            overlaps = bbox_transform.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False))
            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in range(min(boxes.shape[0], gt_boxes.shape[0])):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert gt_ovr >= 0
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert _gt_overlaps[j] == gt_ovr
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps, 'num_pos': num_pos}

    def get_confusion_matrix(self, gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

    def vis_all_mask(self, all_boxes, all_masks, save_path=None):
        """
        visualize all detections in one image
        :param im_array: [b=1 c h w] in rgb
        :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
        :param class_names: list of names in imdb
        :param scale: visualize the scaled image
        :return:
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import random
        import cv2
        from lib.utils.colormap import colormap

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

        color_list = colormap(rgb=True) / 255
        mask_color_id = 0

        for i in range(len(self.roidb)):
            im = np.array(Image.open(self.roidb[i]['image']))
            fig = plt.figure(frameon=False)

            fig.set_size_inches(im.shape[1] / 200, im.shape[0] / 200)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            ax.imshow(im)
            for j, name in enumerate(self.dataset.classes):
                if name == '__background__':
                    continue
                boxes = all_boxes[j][i]
                segms = all_masks[j][i]
                if segms == []:
                    continue
                masks = mask_util.decode(segms)
                for k in range(boxes.shape[0]):
                    score = boxes[k, -1]
                    mask = masks[:, :, k]
                    if score < 0.5:
                        continue
                    bbox = boxes[k, :]
                    ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                      fill=False, edgecolor='g', linewidth=1, alpha=0.5)
                    )
                    ax.text(bbox[0], bbox[1] - 2, name + '{:0.2f}'.format(score).lstrip('0'), fontsize=5, family='serif',
                            bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'), color='white')
                    color_mask = color_list[mask_color_id % len(color_list), 0:3]
                    mask_color_id += 1
                    w_ratio = .4
                    for c in range(3):
                        color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio

                    _, contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    for c in contour:
                        ax.add_patch(
                            Polygon(
                                c.reshape((-1, 2)),
                                fill=True, facecolor=color_mask, edgecolor='w', linewidth=0.8, alpha=0.5
                            )
                        )
            if save_path is None:
                plt.show()
            else:
                fig.savefig(os.path.join(save_path, '{}.png'.format(self.roidb[i]['image'].split('/')[-1])), dpi=200)
            plt.close('all')

    def im_list_to_blob(self, ims, scale=1):
        """Convert a list of images into a network input. Assumes images were
        prepared using prep_im_for_blob or equivalent: i.e.
          - BGR channel order
          - pixel means subtracted
          - resized to the desired input size
          - float32 numpy ndarray format
        Output is a 4D HCHW tensor of the images concatenated along axis 0 with
        shape.
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        # Pad the image so they can be divisible by a stride
        if config.network.has_fpn:
            stride = float(config.network.rpn_feat_stride[-2])
            max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
            max_shape[2] = int(np.ceil(max_shape[2] / stride) * stride)

        num_images = len(ims)
        blob = np.zeros((num_images, 3, int(max_shape[1] * scale), int(max_shape[2] * scale)),
                        dtype=np.float32)
        for i in range(num_images):
            im = ims[i] if scale == 1 else cv2.resize(ims[i].transpose(1, 2, 0), None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
            blob[i, :, 0:im.shape[1], 0:im.shape[2]] = im
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        return blob

    def gt_list_to_blob(self, ims, scale=1):
        """Convert a list of images into a network input. Assumes images were
        prepared using prep_im_for_blob or equivalent: i.e.
          - resized to the desired input size
          - int64 numpy ndarray format
        Output is a 4D HCHW tensor of the images concatenated along axis 0 with
        shape.
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        # Pad the image so they can be divisible by a stride
        if config.network.has_fpn:
            stride = float(config.network.rpn_feat_stride[-2])
            max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
            max_shape[2] = int(np.ceil(max_shape[2] / stride) * stride)

        num_images = ims[0].shape[0]
        blob = np.ones((num_images, int(max_shape[1] * scale), int(max_shape[2] * scale)),
                       dtype=np.int64) * 255
        im = ims[0]
        for i in range(num_images):
            new_im = im[i] if scale == 1 else cv2.resize(im[i], None, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            blob[i, 0:new_im.shape[0], 0:new_im.shape[1]] = new_im
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)

        return blob

    def collate(self, batch):
        if isinstance(batch[0], Sequence):
            transposed = zip(*batch)
            return [self.collate(samples) for samples in transposed]
        elif isinstance(batch[0], int):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], torch.Tensor):
            return torch.cat([b.unsqueeze(0) for b in batch], 0)
        elif batch[0] is None:
            return None
        blob = {}
        for key in batch[0]:
            if key == 'data':
                blob.update({'data': torch.from_numpy(self.im_list_to_blob([b['data'] for b in batch]))})
                if config.network.has_panoptic_head:
                    blob.update({'data_4x': torch.from_numpy(self.im_list_to_blob([b['data'] for b in batch], scale=1/4.))})
            elif key == 'seg_gt':
                blob.update({'seg_gt': torch.from_numpy(self.gt_list_to_blob([b['seg_gt'][np.newaxis, ...] for b in batch]))})
                if config.network.has_panoptic_head:
                    blob.update({'seg_gt_4x': torch.from_numpy(self.gt_list_to_blob([b['seg_gt'][np.newaxis, ...] for b in batch], scale=1/4.))})
            elif key == 'seg_roi_gt':
                assert(len(batch) == 1)
                blob.update({'seg_roi_gt': torch.from_numpy(batch[0]['seg_roi_gt'])})
            elif key == 'mask_gt':
                blob.update({'mask_gt': torch.from_numpy(self.gt_list_to_blob([b['mask_gt'] for b in batch], scale=1./4))})
            elif key == 'im_info':
                blob.update({'im_info': np.vstack([b['im_info'] for b in batch])})
            elif key == 'roidb':
                assert len(batch) == 1
                blob.update({'roidb': batch[0]['roidb']})
            elif key == 'id':
                blob.update({key: torch.cat([torch.from_numpy(np.array([b[key]])) for b in batch], 0)})
            elif key == 'incidence_mat' or key == 'msg_adj':
                blob.update({key: batch[0][key]})
            else:
                blob.update({key: torch.cat([torch.from_numpy(b[key]) for b in batch], 0)})
        return blob
