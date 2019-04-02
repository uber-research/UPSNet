// ---------------------------------------------------------------------------
// Unified Panoptic Segmentation Network
//
// Modifications Copyright (c) 2019 Uber Technologies, Inc.
// ---------------------------------------------------------------------------
// Based on:
// ---------------------------------------------------------------------------
// Caffe2
// Copyright (c) 2017-present, Facebook, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ---------------------------------------------------------------------------



#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cfloat>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

__device__ float bilinear_interpolate(
    const float* bottom_data,
    const int height,
    const int width,
    float y,
    float x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (float)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (float)x_low;
  } else {
    x_high = x_low + 1;
  }

  float ly = y - y_low;
  float lx = x - x_low;
  float hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  float v1 = bottom_data[y_low * width + x_low];
  float v2 = bottom_data[y_low * width + x_high];
  float v3 = bottom_data[y_high * width + x_low];
  float v4 = bottom_data[y_high * width + x_high];
  float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}


__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    float y,
    float x,
    float* w1,
    float* w2,
    float* w3,
    float* w4,
    int* x_low,
    int* x_high,
    int* y_low,
    int* y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    (*w1) = (*w2) = (*w3) = (*w4) = 0.;
    (*x_low) = (*x_high) = (*y_low) = (*y_high) = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  (*y_low) = (int)y;
  (*x_low) = (int)x;

  if ((*y_low) >= height - 1) {
    (*y_high) = (*y_low) = height - 1;
    y = (float)(*y_low);
  } else {
    (*y_high) = (*y_low) + 1;
  }

  if ((*x_low) >= width - 1) {
    (*x_high) = (*x_low) = width - 1;
    x = (float)(*x_low);
  } else {
    (*x_high) = (*x_low) + 1;
  }

  float ly = y - (*y_low);
  float lx = x - (*x_low);
  float hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // float v1 = bottom_data[y_low * width + x_low];
  // float v2 = bottom_data[y_low * width + x_high];
  // float v3 = bottom_data[y_high * width + x_low];
  // float v4 = bottom_data[y_high * width + x_high];
  // float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  (*w1) = hy * hx;
  (*w2) = hy * lx;
  (*w3) = ly * hx;
  (*w4) = ly * lx;

  return;
}

__global__ void RoIAlignForward(
    const int nthreads,
    const float* bottom_data,
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const float* bottom_rois,
    float* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = (int)round(offset_bottom_rois[0]);


    // Do not using rounding; this implementation detail is critical
    float roi_start_w = offset_bottom_rois[1] * spatial_scale;
    float roi_start_h = offset_bottom_rois[2] * spatial_scale;
    float roi_end_w = offset_bottom_rois[3] * spatial_scale;
    float roi_end_h = offset_bottom_rois[4] * spatial_scale;
    // float roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // float roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // float roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // float roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    float roi_width = max(roi_end_w - roi_start_w, (float)1.);
    float roi_height = max(roi_end_h - roi_start_h, (float)1.);
    float bin_size_h = (float)(roi_height) / (float)(pooled_height);
    float bin_size_w = (float)(roi_width) / (float)(pooled_width);

    const float* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    float output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const float y = roi_start_h + ph * bin_size_h +
          (float)(iy + .5f) * bin_size_h /
              (float)(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const float x = roi_start_w + pw * bin_size_w +
            (float)(ix + .5f) * bin_size_w /
                (float)(roi_bin_grid_w);

        float val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}


__global__ void RoIAlignBackwardFeature(
    const int nthreads,
    const float* top_diff,
    const int num_rois,
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    float* bottom_diff,
    const float* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = (int)round(offset_bottom_rois[0]);

    // Do not using rounding; this implementation detail is critical
    float roi_start_w = offset_bottom_rois[1] * spatial_scale;
    float roi_start_h = offset_bottom_rois[2] * spatial_scale;
    float roi_end_w = offset_bottom_rois[3] * spatial_scale;
    float roi_end_h = offset_bottom_rois[4] * spatial_scale;
    // float roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // float roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // float roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // float roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    float roi_width = max(roi_end_w - roi_start_w, (float)1.);
    float roi_height = max(roi_end_h - roi_start_h, (float)1.);
    float bin_size_h = (float)(roi_height) / (float)(pooled_height);
    float bin_size_w = (float)(roi_width) / (float)(pooled_width);

    float* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const float* offset_top_diff = top_diff + top_offset;
    const float top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const float y = roi_start_h + ph * bin_size_h +
          (float)(iy + .5f) * bin_size_h /
              (float)(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const float x = roi_start_w + pw * bin_size_w +
            (float)(ix + .5f) * bin_size_w /
                (float)(roi_bin_grid_w);

        float w1 = 0, w2 = 0, w3 = 0, w4 = 0;
        int x_low = 0, x_high = 0, y_low = 0, y_high = 0;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            &w1,
            &w2,
            &w3,
            &w4,
            &x_low,
            &x_high,
            &y_low,
            &y_high,
            index);

        float g1 = top_diff_this_bin * w1 / count;
        float g2 = top_diff_this_bin * w2 / count;
        float g3 = top_diff_this_bin * w3 / count;
        float g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          gpu_atomic_add(
             (float)(g1), offset_bottom_diff + y_low * width + x_low);
          gpu_atomic_add(
             (float)(g2), offset_bottom_diff + y_low * width + x_high);
          gpu_atomic_add(
             (float)(g3), offset_bottom_diff + y_high * width + x_low);
          gpu_atomic_add(
             (float)(g4), offset_bottom_diff + y_high * width + x_high);
          // atomicAdd(bottom_diff + y_low * width + x_low, g1);
          // atomicAdd(bottom_diff + y_low * width + x_high, g2);
          // atomicAdd(bottom_diff + y_high * width + x_low, g3);
          // atomicAdd(bottom_diff + y_high * width + x_high, g4);
          // atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
          // atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
          // atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
          // atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
        } // if
      } // ix
    } // iy
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward


int roi_align_forward_gpu_kernel_launcher(
  cudaStream_t stream, const float* bottom_data, const float spatial_scale, 
  const int num_rois, const int height, const int width, const int channels, 
  const int pooled_height, const int pooled_width, const int sampling_ratio, 
  const float* bottom_rois, float* top_data) {

  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  cudaError_t err;

  RoIAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
    output_size, bottom_data, spatial_scale, channels, height, width, pooled_height,
    pooled_width, sampling_ratio, bottom_rois, top_data);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
      exit( -1 );
  }

  return 1;
}

int roi_align_backward_gpu_kernel_launcher(
  cudaStream_t stream, const float* top_diff, const float spatial_scale, 
  const int batch_size, const int num_rois, const int height, const int width, 
  const int channels, const int pooled_height, const int pooled_width, 
  const int sampling_ratio, const float* bottom_rois, float* bottom_diff) {

  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * channels * pooled_height * pooled_width;
  cudaError_t err;


  RoIAlignBackwardFeature<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
    output_size, top_diff, num_rois, spatial_scale, channels, height, width, pooled_height,
    pooled_width, sampling_ratio, bottom_diff, bottom_rois);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
      exit( -1 );
  }

  return 1;
}
