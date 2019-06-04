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

import argparse
from .config import config, update_config

def parse_args(description=''):
    parser = argparse.ArgumentParser(description=description)
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--eval_only', help='if only eval existing results', action='store_true')
    parser.add_argument('--weight_path', help='manually specify model weights', type=str, default='')

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    args = parser.parse_args()
    return args
