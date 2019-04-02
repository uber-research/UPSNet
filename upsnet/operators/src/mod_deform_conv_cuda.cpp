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

void modulated_deformable_im2col_gpu_kernel_launcher(cudaStream_t stream,
  const float *data_im, const float *data_offset, const float *data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im,
  const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int deformable_group, float *data_col);

void modulated_deformable_col2im_gpu_kernel_launcher(cudaStream_t stream,
  const float *data_col, const float *data_offset, const float *data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im,
  const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int deformable_group, float *grad_im);

void modulated_deformable_col2im_coord_gpu_kernel_launcher(cudaStream_t stream,
  const float *data_col, const float *data_im, const float *data_offset, const float *data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im,
  const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int deformable_group,
  float *grad_offset, float *grad_mask);


int modulated_deformable_im2col_cuda(
  at::Tensor data_im_cuda, at::Tensor data_offset_cuda, at::Tensor data_mask_cuda,
  vector<int> im_shape, vector<int> col_shape, vector<int> kernel_shape, vector<int> pad, vector<int> stride, vector<int> dilation,
  const uint32_t deformable_group, at::Tensor data_col_cuda) {

  cudaStream_t stream = THCState_getCurrentStream(state);

  float* data_im     = data_im_cuda.data<float>();
  float* data_offset = data_offset_cuda.data<float>();
  float* data_mask   = data_mask_cuda.data<float>();
  float* data_col    = data_col_cuda.data<float>();

  modulated_deformable_im2col_gpu_kernel_launcher(
    stream, data_im, data_offset, data_mask, 1, im_shape[1], im_shape[2], im_shape[3], col_shape[1], col_shape[2], kernel_shape[0], kernel_shape[1],
    pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1], deformable_group, data_col);

  return 1;
}

int modulated_deformable_col2im_cuda(
  at::Tensor data_col_cuda, at::Tensor data_offset_cuda, at::Tensor data_mask_cuda,
  vector<int> im_shape, vector<int> col_shape, vector<int> kernel_shape, vector<int> pad, vector<int> stride, vector<int> dilation,
  const uint32_t deformable_group, at::Tensor grad_im_cuda) {

  cudaStream_t stream = THCState_getCurrentStream(state);

  float* data_col    = data_col_cuda.data<float>();
  float* data_offset = data_offset_cuda.data<float>();
  float* data_mask   = data_mask_cuda.data<float>();
  float* grad_im     = grad_im_cuda.data<float>();

  modulated_deformable_col2im_gpu_kernel_launcher(
    stream, data_col, data_offset, data_mask, 1, im_shape[1], im_shape[2], im_shape[3], col_shape[1], col_shape[2], kernel_shape[0], kernel_shape[1], 
    pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1], deformable_group, grad_im);

  return 1;
}

int modulated_deformable_col2im_coord_cuda(
  at::Tensor data_col_cuda, at::Tensor data_im_cuda, at::Tensor data_offset_cuda, at::Tensor data_mask_cuda,
  vector<int> im_shape, vector<int> col_shape, vector<int> kernel_shape, vector<int> pad, vector<int> stride, vector<int> dilation,
  const uint32_t deformable_group, at::Tensor grad_offset_cuda, at::Tensor grad_mask_cuda) {

  cudaStream_t stream = THCState_getCurrentStream(state);

  float* data_col    = data_col_cuda.data<float>();
  float* data_im     = data_im_cuda.data<float>();
  float* data_offset = data_offset_cuda.data<float>();
  float* data_mask   = data_mask_cuda.data<float>();
  float* grad_offset = grad_offset_cuda.data<float>();
  float* grad_mask   = grad_mask_cuda.data<float>();

  modulated_deformable_col2im_coord_gpu_kernel_launcher(
    stream, data_col, data_im, data_offset, data_mask, 1, im_shape[1], im_shape[2], im_shape[3], col_shape[1], col_shape[2], kernel_shape[0], kernel_shape[1], 
    pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1], deformable_group, grad_offset, grad_mask);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mod_deform_im2col", &modulated_deformable_im2col_cuda, "Modulated Deformable Convolution im2col (CUDA)");
    m.def("mod_deform_col2im", &modulated_deformable_col2im_cuda, "Modulated Deformable Convolution col2im (CUDA)");
    m.def("mod_deform_col2im_coord", &modulated_deformable_col2im_coord_cuda, "Modulated Deformable Convolution col2im coordinate (CUDA)");
}