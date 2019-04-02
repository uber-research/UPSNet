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

import time
import logging

class Speedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, count, metrics):
        """Callback to Show speed."""
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if metrics is not None:
                    s = "Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (count, speed)
                    for metric in metrics:
                        s += "%s=%f,\t" % (metric.get())
                else:
                    s = "Batch [%d]\tSpeed: %.2f samples/sec" % (count, speed)

                logging.info(s)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()
