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
// 
// Written by Yuwen Xiong
// ---------------------------------------------------------------------------

#include <THC/THC.h>
#include <math.h>
#include <torch/torch.h>
#include <vector>

using std::vector;

extern THCState *state;


void roi_align_forward_gpu_kernel_launcher(cudaStream_t stream,
  const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
  const int width, const int channels, const int pooled_height,
  const int pooled_width, const int sampling_ratio, const float* bottom_rois,
  float* top_data);

void roi_align_backward_gpu_kernel_launcher(cudaStream_t stream, 
  const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
  const int height, const int width, const int channels, const int pooled_height,
  const int pooled_width, const int sampling_ratio, const float* bottom_rois,
  float* bottom_diff);


int roi_align_forward_cuda(
  int pooled_height, int pooled_width, int sampling_ratio, float spatial_scale,
  at::Tensor features, at::Tensor rois, at::Tensor output) {

  // Grab the input tensor
  float* data_flat = features.data<float>();
  float* rois_flat = rois.data<float>();

  float* output_flat = output.data<float>();

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 5)
  {
      return 0;
  }

  // batch size
  int batch_size = features.size(0);
  // data height
  int data_height = features.size(2);
  // data width
  int data_width = features.size(3);
  // Number of channels
  int num_channels = features.size(1);

  cudaStream_t stream = THCState_getCurrentStream(state);

  roi_align_forward_gpu_kernel_launcher(
      stream, data_flat, spatial_scale, num_rois, 
      data_height, data_width, num_channels, 
      pooled_height, pooled_width, sampling_ratio, 
      rois_flat, output_flat);

  return 1;    
}

int roi_align_backward_cuda(
  int pooled_height, int pooled_width, int sampling_ratio, float spatial_scale,
  at::Tensor top_grad, at::Tensor rois, at::Tensor bottom_grad) {
  // Grab the input tensor
  float* top_grad_flat = top_grad.data<float>();
  float* rois_flat = rois.data<float>();

  float* bottom_grad_flat = bottom_grad.data<float>();

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 5)
  {
      return 0;
  }

  // batch size
  int batch_size = bottom_grad.size(0);
  // data height
  int data_height = bottom_grad.size(2);
  // data width
  int data_width = bottom_grad.size(3);
  // Number of channels
  int num_channels = bottom_grad.size(1);

  cudaStream_t stream = THCState_getCurrentStream(state);
  roi_align_backward_gpu_kernel_launcher(
      stream, top_grad_flat, spatial_scale, batch_size,
      num_rois, data_height, data_width, num_channels, 
      pooled_height, pooled_width, sampling_ratio, 
      rois_flat, bottom_grad_flat);

  return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("roi_align_forward", &roi_align_forward_cuda, "RoI Align forward (CUDA)");
    m.def("roi_align_backward", &roi_align_backward_cuda, "RoI Align backward (CUDA)");
}