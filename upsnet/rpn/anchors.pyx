# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Modifications Copyright (c) 2019 Uber Technologies, Inc.
# ---------------------------------------------------------------------------
# Based on:
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


cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def anchors_cython(int height, int width, int stride, np.ndarray[DTYPE_t, ndim=2] base_anchors):
    """
    Parameters
    ----------
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: (A, 4) a base set of anchors
    Returns
    -------
    all_anchors: (height, width, A, 4) ndarray of anchors spreading over the plane
    """
    cdef unsigned int A = base_anchors.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=4] all_anchors = np.zeros((height, width, A, 4), dtype=DTYPE)
    cdef unsigned int iw, ih
    cdef unsigned int k
    cdef unsigned int sh
    cdef unsigned int sw
    for iw in range(width):
        sw = iw * stride
        for ih in range(height):
            sh = ih * stride
            for k in range(A):
                all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + sw
                all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + sh
                all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + sw
                all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + sh
    return all_anchors
