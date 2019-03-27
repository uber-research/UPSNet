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

from __future__ import print_function, division
import os
import sys
import logging
import pprint
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import tensorboardX
import cv2
import torch.utils.data.distributed as distributed


sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collections import deque
from upsnet.config.config import config
from upsnet.config.parse_args import parse_args
from lib.utils.logging import create_logger

args = parse_args()

if config.train.use_horovod:
    import horovod.torch as hvd
    from horovod.torch.mpi_ops import allreduce_async
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

is_master = (not config.train.use_horovod) or hvd.rank() == 0

if is_master:
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(config.output_path, 'tensorboard',
                                                             os.path.basename(args.cfg).split('.')[0],
                                                             '_'.join(config.dataset.image_set.split('+')),
                                                             time.strftime('%Y-%m-%d-%H-%M')))
else:
    final_output_path = os.path.join(config.output_path, os.path.basename(args.cfg).split('.')[0], '{}'.format('_'.join([iset for iset in config.dataset.image_set.split('+')])))

from upsnet.dataset import *
from upsnet.models import *
from lib.utils.callback import Speedometer
from lib.utils.data_parallel import DataParallel
from lib.utils.metric import AvgMetric
from lib.nn.optimizer import SGD, Adam, clip_grad

np.random.seed(235)
torch.cuda.manual_seed_all(235)
torch.manual_seed(235)

cv2.ocl.setUseOpenCL(False)
cudnn.enabled = True
cudnn.benchmark = False


def lr_poly(base_lr, iter, max_iter, warmup_iter=0):
    power = 0.9
    if iter < warmup_iter:
        alpha = iter / warmup_iter
        return min(base_lr * (1 / 10.0 * (1 - alpha) + alpha), base_lr * ((1 - float(iter) / max_iter)**(power)))
    return base_lr * ((1 - float(iter) / max_iter)**(power))


def get_step_index(iter, decay_iters):
    for idx, decay_iter in enumerate(decay_iters):
        if iter < decay_iter:
            return idx
    return len(decay_iters)


def lr_factor(base_lr, iter, decay_iter, warmup_iter=0):
    if iter < warmup_iter:
        alpha = iter / warmup_iter
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)
    return base_lr * (0.1 ** get_step_index(iter, decay_iter))


def adjust_learning_rate(optimizer, iter, config):
    assert config.train.lr_schedule in ['step', 'poly']
    if config.train.lr_schedule == 'step':
        return lr_factor(config.train.lr, iter, config.train.decay_iteration, config.train.warmup_iteration)
    if config.train.lr_schedule == 'poly':
        return lr_poly(config.train.lr, iter, config.train.max_iteration, config.train.warmup_iteration)

def upsnet_train():

    if is_master:
        logger.info('training config:{}\n'.format(pprint.pformat(config)))
    gpus = [torch.device('cuda', int(_)) for _ in config.gpus.split(',')]
    num_replica = hvd.size() if config.train.use_horovod else len(gpus)
    num_gpus = 1 if config.train.use_horovod else len(gpus)

    # create models
    train_model = eval(config.symbol)().cuda()
        
    # create optimizer
    params_lr = train_model.get_params_lr()
    # we use custom optimizer and pass lr=1 to support different lr for different weights
    optimizer = SGD(params_lr, lr=1, momentum=config.train.momentum, weight_decay=config.train.wd)
    if config.train.use_horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=train_model.named_parameters())
    optimizer.zero_grad()

    # create data loader
    train_dataset = eval(config.dataset.dataset)(image_sets=config.dataset.image_set.split('+'), flip=config.train.flip, result_path=final_output_path)
    val_dataset = eval(config.dataset.dataset)(image_sets=config.dataset.test_image_set.split('+'), flip=False, result_path=final_output_path, phase='val')
    if config.train.use_horovod:
        train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        val_sampler = distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size, sampler=train_sampler, num_workers=num_gpus * 4, drop_last=False, collate_fn=train_dataset.collate)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size, sampler=val_sampler, num_workers=num_gpus * 4, drop_last=False, collate_fn=val_dataset.collate)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=config.train.shuffle, num_workers=num_gpus * 4 if not config.debug_mode else num_gpus * 4, drop_last=False, collate_fn=train_dataset.collate)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=num_gpus * 4 if not config.debug_mode else num_gpus * 4, drop_last=False, collate_fn=val_dataset.collate)

    # preparing
    curr_iter = config.train.begin_iteration
    batch_end_callback = [Speedometer(num_replica * config.train.batch_size, config.train.display_iter)]
    metrics = []
    metrics_name = []
    if config.network.has_rpn:
        metrics.extend([AvgMetric(name='rpn_cls_loss'), AvgMetric(name='rpn_bbox_loss'),])
        metrics_name.extend(['rpn_cls_loss', 'rpn_bbox_loss'])
    if config.network.has_rcnn:
        metrics.extend([AvgMetric(name='rcnn_accuracy'), AvgMetric(name='cls_loss'), AvgMetric(name='bbox_loss'),])
        metrics_name.extend(['rcnn_accuracy', 'cls_loss', 'bbox_loss'])
    if config.network.has_mask_head:
        metrics.extend([AvgMetric(name='mask_loss'), ])
        metrics_name.extend(['mask_loss'])
    if config.network.has_fcn_head:
        metrics.extend([AvgMetric(name='fcn_loss'), ])
        metrics_name.extend(['fcn_loss'])
        if config.train.fcn_with_roi_loss:
            metrics.extend([AvgMetric(name='fcn_roi_loss'), ])
            metrics_name.extend(['fcn_roi_loss'])
    if config.network.has_panoptic_head:
        metrics.extend([AvgMetric(name='panoptic_accuracy'), AvgMetric(name='panoptic_loss'), ])
        metrics_name.extend(['panoptic_accuracy', 'panoptic_loss'])

    if config.train.resume:
        train_model.load_state_dict(torch.load(os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.pth')), resume=True)
        optimizer.load_state_dict(torch.load(os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.state.pth')))
        if config.train.use_horovod:
            hvd.broadcast_parameters(train_model.state_dict(), root_rank=0)
    else:
        if is_master:
            train_model.load_state_dict(torch.load(config.network.pretrained))

        if config.train.use_horovod:
            hvd.broadcast_parameters(train_model.state_dict(), root_rank=0)

    if not config.train.use_horovod:
        train_model = DataParallel(train_model, device_ids=[int(_) for _ in config.gpus.split(',')]).to(gpus[0])

    if is_master:
        batch_end_callback[0](0, 0)

    # start training
    while curr_iter < config.train.max_iteration:
        if config.train.use_horovod:
            train_sampler.set_epoch(curr_iter)

            train_model.eval() # freeze bn layer

            for inner_iter, batch in enumerate(train_loader):
                data, label, _ = batch
                for k, v in data.items():
                    data[k] = v if not torch.is_tensor(v) else v.cuda()
                for k, v in label.items():
                    label[k] = v if not torch.is_tensor(v) else v.cuda()

                lr = adjust_learning_rate(optimizer, curr_iter, config)
                optimizer.zero_grad()
                output = train_model(data, label)
                loss = 0
                if config.network.has_rpn:
                    loss = loss + output['rpn_cls_loss'].mean() + output['rpn_bbox_loss'].mean()
                if config.network.has_rcnn:
                    loss = loss + output['cls_loss'].mean() + output['bbox_loss'].mean() * config.train.bbox_loss_weight
                if config.network.has_mask_head:
                    loss = loss + output['mask_loss'].mean()
                if config.network.has_fcn_head:
                    loss = loss + output['fcn_loss'].mean() * config.train.fcn_loss_weight
                    if config.train.fcn_with_roi_loss:
                        loss = loss + output['fcn_roi_loss'].mean() * config.train.fcn_loss_weight * 0.2
                if config.network.has_panoptic_head:
                    loss = loss + output['panoptic_loss'].mean() * config.train.panoptic_loss_weight
                loss.backward()
                optimizer.step(lr)

                losses = []
                losses.append(allreduce_async(loss, name='train_total_loss'))
                for l in metrics_name:
                    losses.append(allreduce_async(output[l].mean(), name=l))

                loss = hvd.synchronize(losses[0]).item()
                if is_master:
                    writer.add_scalar('train_total_loss', loss, curr_iter)
                for i, (metric, l) in enumerate(zip(metrics, metrics_name)):
                    loss = hvd.synchronize(losses[i + 1]).item()
                    if is_master:
                        writer.add_scalar('train_' + l, loss, curr_iter)
                        metric.update(_, _, loss)
                curr_iter += 1


                if curr_iter in config.train.decay_iteration:
                    if is_master:
                        logger.info('decay momentum buffer')
                    for k in optimizer.state_dict()['state'].keys():
                        if 'momentum_buffer' in optimizer.state_dict()['state'][k]:
                            optimizer.state_dict()['state'][k]['momentum_buffer'].div_(10)

                if is_master:
                    if curr_iter % config.train.display_iter == 0:
                        for callback in batch_end_callback:
                            callback(curr_iter, metrics)

                    if curr_iter % config.train.snapshot_step == 0:
                        logger.info('taking snapshot ...')
                        torch.save(train_model.state_dict(), os.path.join(final_output_path, config.model_prefix+str(curr_iter)+'.pth'))
                        torch.save(optimizer.state_dict(), os.path.join(final_output_path, config.model_prefix+str(curr_iter)+'.state.pth'))
        else:
            inner_iter = 0
            train_iterator = train_loader.__iter__()
            while inner_iter + num_gpus <= len(train_loader):
                batch = []
                for gpu_id in gpus:
                    data, label, _ = train_iterator.next()
                    for k, v in data.items():
                        data[k] = v if not torch.is_tensor(v) else v.pin_memory().to(gpu_id, non_blocking=True)
                    for k, v in label.items():
                        label[k] = v if not torch.is_tensor(v) else v.pin_memory().to(gpu_id, non_blocking=True)
                    batch.append((data, label))
                    inner_iter += 1
                lr = adjust_learning_rate(optimizer, curr_iter, config)
                optimizer.zero_grad()
                if config.train.use_horovod:
                    output = train_model(data, label)
                else:
                    output = train_model(*batch)

                loss = 0
                if config.network.has_rpn:
                    loss = loss + output['rpn_cls_loss'].mean() + output['rpn_bbox_loss'].mean()
                if config.network.has_rcnn:
                    loss = loss + output['cls_loss'].mean() + output['bbox_loss'].mean()
                if config.network.has_mask_head:
                    loss = loss + output['mask_loss'].mean()
                if config.network.has_fcn_head:
                    loss = loss + output['fcn_loss'].mean() * config.train.fcn_loss_weight
                    if config.train.fcn_with_roi_loss:
                        loss = loss + output['fcn_roi_loss'].mean() * config.train.fcn_loss_weight * 0.2
                if config.network.has_panoptic_head:
                    loss = loss + output['panoptic_loss'].mean() * config.train.panoptic_loss_weight
                loss.backward()
                optimizer.step(lr)
                
                losses = []
                losses.append(loss.item())
                for l in metrics_name:
                    losses.append(output[l].mean().item())

                loss = losses[0]
                if is_master:
                    writer.add_scalar('train_total_loss', loss, curr_iter)
                for i, (metric, l) in enumerate(zip(metrics, metrics_name)):
                    loss = losses[i + 1]
                    if is_master:
                        writer.add_scalar('train_' + l, loss, curr_iter)
                        metric.update(_, _, loss)
                curr_iter += 1

                if curr_iter in config.train.decay_iteration:
                    if is_master:
                        logger.info('decay momentum buffer')
                    for k in optimizer.state_dict()['state'].keys():
                        optimizer.state_dict()['state'][k]['momentum_buffer'].div_(10)

                if is_master:
                    if curr_iter % config.train.display_iter == 0:
                        for callback in batch_end_callback:
                            callback(curr_iter, metrics)


                    if curr_iter % config.train.snapshot_step == 0:
                        logger.info('taking snapshot ...')
                        torch.save(train_model.module.state_dict(), os.path.join(final_output_path, config.model_prefix+str(curr_iter)+'.pth'))
                        torch.save(optimizer.state_dict(), os.path.join(final_output_path, config.model_prefix+str(curr_iter)+'.state.pth'))

            while True:
                try:
                    train_iterator.next()
                except:
                    break

        for metric in metrics:
            metric.reset()

        if config.train.eval_data:
            train_model.eval()

            if config.train.use_horovod:
                for inner_iter, batch in enumerate(val_loader):
                    data, label, _ = batch
                    for k, v in data.items():
                        data[k] = v if not torch.is_tensor(v) else v.cuda(non_blocking=True)
                    for k, v in label.items():
                        label[k] = v if not torch.is_tensor(v) else v.cuda(non_blocking=True)

                    with torch.no_grad():
                        output = train_model(data, label)

                    for metric, l in zip(metrics, metrics_name):
                        loss = hvd.allreduce(output[l].mean()).item()
                        if is_master:
                            metric.update(_, _, loss)

            else:
                inner_iter = 0
                val_iterator = val_loader.__iter__()
                while inner_iter + len(gpus) <= len(val_loader):
                    batch = []
                    for gpu_id in gpus:
                        data, label, _ = val_iterator.next()
                        for k, v in data.items():
                            data[k] = v if not torch.is_tensor(v) else v.pin_memory().to(gpu_id, non_blocking=True)
                        for k, v in label.items():
                            label[k] = v if not torch.is_tensor(v) else v.pin_memory().to(gpu_id, non_blocking=True)
                        batch.append((data, label))
                        inner_iter += 1

                    with torch.no_grad():
                        if config.train.use_horovod:
                            output = train_model(data, label)
                        else:
                            output = train_model(*batch)

                    losses = []
                    for l in metrics_name:
                        losses.append(allreduce_async(output[l].mean(), name=l) if config.train.use_horovod else output[l].mean().item())

                    for metric, loss in zip(metrics, losses):
                        loss = hvd.synchronize(loss).item() if config.train.use_horovod else loss
                        if is_master:
                            metric.update(_, _, loss)

                while True:
                    try:
                        val_iterator.next()
                    except Exception:
                        break

            s = 'Batch [%d]\t Epoch[%d]\t' % (curr_iter, curr_iter // len(train_loader) // num_gpus)

            for metric in metrics:
                m, v = metric.get()
                s += 'Val-%s=%f,\t' % (m, v)
                if is_master:
                    writer.add_scalar('val_' + m, v, curr_iter)
                    metric.reset()
            if is_master:
                logger.info(s)

    if is_master and config.train.use_horovod:
        logger.info('taking snapshot ...')
        torch.save(train_model.state_dict(), os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.state.pth'))
    elif not config.train.use_horovod:
        logger.info('taking snapshot ...')
        torch.save(train_model.module.state_dict(), os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(final_output_path, config.model_prefix + str(curr_iter) + '.state.pth'))

if __name__ == '__main__':
    upsnet_train()
