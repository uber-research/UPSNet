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

import os
import torch
from functools import reduce
from itertools import accumulate
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from subprocess import call

def _create_module_dir(base_path, fullname):
    module, _, name = fullname.rpartition('.')
    if not module:
        target_dir = name
    else:
        target_dir = reduce(os.path.join, fullname.split('.'))
    target_dir = os.path.join(base_path, target_dir)
    try:
        os.makedirs(target_dir)
    except os.error:
        pass
    for dirname in accumulate(fullname.split('.'), os.path.join):
        init_file = os.path.join(base_path, dirname, '__init__.py')
        open(init_file, 'a').close()  # Create file if it doesn't exist yet
    return name, target_dir

base_path = os.path.abspath(os.path.dirname('.'))
_create_module_dir(base_path, '_ext.deform_conv')
setup(
    name='mod_deform_conv',
    ext_modules=[
        CUDAExtension('mod_deform_conv_cuda', [
            'src/mod_deform_conv_cuda.cpp',
            'src/mod_deform_conv_kernel.cu',
        ],
            include_dirs=[os.path.join(base_path, 'src')]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

call('mv mod_deform_conv_cuda*.so _ext/mod_deform_conv/', shell=True)
