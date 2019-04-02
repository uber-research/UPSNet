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
import torch.nn.functional as F

class EvalMetric(object):
    """Base class for all evaluation metrics.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, name, output_names=None,
                 label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "{}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(**config)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    # def update_dict(self, label, pred):
    #     """Update the internal evaluation with named label and pred
    #
    #     Parameters
    #     ----------
    #     labels : OrderedDict of str -> NDArray
    #         name to array mapping for labels.
    #
    #     preds : list of NDArray
    #         name to array mapping of predicted outputs.
    #     """
    #     if self.output_names is not None:
    #         pred = [pred[name] for name in self.output_names]
    #     else:
    #         pred = list(pred.values())
    #
    #     if self.label_names is not None:
    #         label = [label[name] for name in self.label_names]
    #     else:
    #         label = list(label.values())
    #
    #     self.update(label, pred)

    def update(self, preds, labels, loss):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

class AvgMetric(EvalMetric):
    def __init__(self, scale=1, name="AverageScalar"):
        super(AvgMetric, self).__init__(name)
        self.scale = scale
        self.reset()

    def update(self, predict, target, loss):
        self.sum_metric += loss / self.scale
        self.num_inst += 1

    def reset(self):
        self.sum_metric = 0
        self.num_inst = 0

class AccWithIgnoreMetric(EvalMetric):
    def __init__(self, ignore_label=255, name="AccWithIgnore"):
        super(AccWithIgnoreMetric, self).__init__(name)
        self.ignore_label = ignore_label
        self.reset()

    def update(self, predict, target, loss):
        assert predict.dim() == 4
        assert target.dim() == 3
        _, predict_label = predict.data.max(dim=1)
        predict_label = predict_label.int()
        target = target.data.int()
        self.sum_metric += (predict_label.view(-1) == target.view(-1)).sum()
        self.num_inst += len(predict_label.view(-1)) - (target.view(-1) == self.ignore_label).sum()

    def reset(self):
        self.sum_metric = 0
        self.num_inst = 0

class IoUMetric(EvalMetric):
    def __init__(self, label_num, ignore_label=255, name="IoU"):
        self.label_num = label_num
        self.ignore_label = ignore_label
        super(IoUMetric, self).__init__(name)

    def reset(self):
        self._tp = [0.0] * self.label_num
        self._denom = [0.0] * self.label_num

    def update(self, predict, target, loss):
        assert predict.dim() == 4
        assert target.dim() == 3
        _, predict_label = predict.data.max(dim=1)
        predict_label = predict_label.int()
        target = target.data.int()
        assert len(predict_label.size()) == 3

        for i in range(target.size(0)):
            label = target[i, :, :]
            pred_label = predict_label[i, :, :]

            iou = 0
            eps = 1e-6
            # skip_label_num = 0
            for j in range(self.label_num):
                pred_cur = (pred_label.view(-1) == j)
                gt_cur = (label.view(-1) == j)
                tp = (pred_cur & gt_cur).sum()
                denom = (pred_cur | gt_cur).sum() - (pred_cur & (label.view(-1) == self.ignore_label)).sum()
                assert tp <= denom
                self._tp[j] += tp
                self._denom[j] += denom
                iou += self._tp[j] / (self._denom[j] + eps)
            iou /= self.label_num
            self.sum_metric = iou
            self.num_inst = 1
