// ---------------------------------------------------------------------------
// Unified Panoptic Segmentation Network
// 
// Copyright (c) 2018-2019 Uber Technologies, Inc.
// 
// Licensed under the Uber Non-Commercial License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at the root directory of this project. 
// 
// See the License for the specific language governing permissions and
// limitations under the License.
// ---------------------------------------------------------------------------

void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);
