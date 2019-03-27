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

void deformable_im2col_gpu_kernel_launcher(cudaStream_t stream,
  const float *data_im, const float *data_offset, const int channels,
  const int height, const int width, const int ksize_h, const int ksize_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w, const int parallel_imgs,
  const int deformable_group, float *data_col);


void deformable_col2im_gpu_kernel_launcher(cudaStream_t stream,
  const float *data_col, const float *data_offset, const int channels,
  const int height, const int width, const int ksize_h,
  const int ksize_w, const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int parallel_imgs, const int deformable_group,
  float* grad_im);

void deformable_col2im_coord_gpu_kernel_launcher(cudaStream_t stream,
  const float *data_col, const float *data_im, const float *data_offset, const int channels,
  const int height, const int width, const int ksize_h, const int ksize_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w, const int parallel_imgs,
  const int deformable_group, float *grad_offset);

int deformable_im2col_cuda(
  at::Tensor data_im_cuda, at::Tensor data_offset_cuda,
  vector<int> im_shape, vector<int> col_shape, vector<int> kernel_shape, vector<int> pad, vector<int> stride, vector<int> dilation,
  const uint32_t parallel_imgs, const uint32_t deformable_group, at::Tensor data_col_cuda) {

  cudaStream_t stream = THCState_getCurrentStream(state);

  float* data_im     = data_im_cuda.data<float>();
  float* data_offset = data_offset_cuda.data<float>();
  float* data_col    = data_col_cuda.data<float>();

  deformable_im2col_gpu_kernel_launcher(
    stream, data_im, data_offset, im_shape[1], im_shape[2], im_shape[3], kernel_shape[0], kernel_shape[1],
    pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1], parallel_imgs, deformable_group, data_col);

  // deformable_im2col_gpu_kernel_launcher(
  //   stream, data_im, data_offset, im_shape, col_shape, kernel_shape, pad, stride, dilation, 1, deformable_group, data_col);

  return 1;
}

int deformable_col2im_cuda(
  at::Tensor data_col_cuda, at::Tensor data_offset_cuda,
  vector<int> im_shape, vector<int> col_shape, vector<int> kernel_shape, vector<int> pad, vector<int> stride,
  vector<int> dilation, const uint32_t parallel_imgs, const uint32_t deformable_group, at::Tensor grad_im_cuda) {

  cudaStream_t stream = THCState_getCurrentStream(state);

  float* data_col    = data_col_cuda.data<float>();
  float* data_offset = data_offset_cuda.data<float>();
  float* grad_im     = grad_im_cuda.data<float>();

  deformable_col2im_gpu_kernel_launcher(
    stream, data_col, data_offset, im_shape[1], im_shape[2], im_shape[3], kernel_shape[0], kernel_shape[1], 
    pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1], parallel_imgs, deformable_group, grad_im);

  return 1;
}

int deformable_col2im_coord_cuda(
  at::Tensor data_col_cuda, at::Tensor data_im_cuda, at::Tensor data_offset_cuda,
  vector<int> im_shape, vector<int> col_shape, vector<int> kernel_shape, vector<int> pad, vector<int> stride,
  vector<int> dilation, const uint32_t parallel_imgs, const uint32_t deformable_group, at::Tensor grad_offset_cuda) {

  cudaStream_t stream = THCState_getCurrentStream(state);

  float* data_col    = data_col_cuda.data<float>();
  float* data_im     = data_im_cuda.data<float>();
  float* data_offset = data_offset_cuda.data<float>();
  float* grad_offset = grad_offset_cuda.data<float>();

  deformable_col2im_coord_gpu_kernel_launcher(
    stream, data_col, data_im, data_offset, im_shape[1], im_shape[2], im_shape[3], kernel_shape[0], kernel_shape[1], 
    pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1], parallel_imgs, deformable_group, grad_offset);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("deform_im2col", &deformable_im2col_cuda, "Deformable Convolution im2col (CUDA)");
    m.def("deform_col2im", &deformable_col2im_cuda, "Deformable Convolution col2im (CUDA)");
    m.def("deform_col2im_coord", &deformable_col2im_coord_cuda, "Deformable Convolution col2im coordinate (CUDA)");
}