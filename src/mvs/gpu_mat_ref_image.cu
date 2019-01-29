// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "mvs/gpu_mat_ref_image.h"

#include <iostream>
#include <stdio.h>

#include "util/cudacc.h"

namespace colmap {
namespace mvs {
namespace {

// reference image texture is W x H x C float typed texture
texture<float, cudaTextureType2DLayered> image_texture;
__global__ void FilterKernel(GpuMat<float> image, GpuMat<float> sum_image,
                             GpuMat<float> squared_sum_image,
                             const int window_radius, const int window_step,
                             const float sigma_spatial,
                             const float sigma_color) {
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int channel = blockDim.z * blockIdx.z + threadIdx.z;
  if (row >= image.GetHeight() || col >= image.GetWidth() ||
      channel >= image.GetDepth()) {
    return;
  }
  MultiChannelWeightComputer multi_channel_weight_computer_(sigma_spatial,
                                                            sigma_color);

  const float center_feature = tex2DLayered(image_texture, col, row, channel);
  float feature_sum = 0.0f;
  float feature_squared_sum = 0.0f;
  float bilateral_weight_sum = 0.0f;

  for (int window_row = -window_radius; window_row <= window_radius;
       window_row += window_step) {
    for (int window_col = -window_radius; window_col <= window_radius;
         window_col += window_step) {
      const float feature =
        tex2DLayered(image_texture, col + window_col, row + window_row, channel);
      const float bilateral_weight = multi_channel_weight_computer_.Compute(
        window_row, window_col, center_feature, feature);
      feature_sum += bilateral_weight * feature;
      feature_squared_sum += bilateral_weight * feature * feature;
      bilateral_weight_sum += bilateral_weight;
    }
  }
  /* printf("col: %d, row: %d, channel:  %d, center_feature: %f, sum: %f", */
  /*        col, row, channel, center_feature, feature_sum); */

  feature_sum /= bilateral_weight_sum;
  feature_squared_sum /= bilateral_weight_sum;

  image.Set(row, col, channel, center_feature);
  sum_image.Set(row, col, channel, feature_sum);
  squared_sum_image.Set(row, col, channel, feature_squared_sum);
}

}  // namespace

// Adding Channel as input in constructor
GpuMatRefImage::GpuMatRefImage(const size_t width, const size_t height,
                               const size_t channel)
    : height_(height), width_(width), channel_(channel){
  image.reset(new GpuMat<float>(width, height, channel));
  sum_image.reset(new GpuMat<float>(width, height, channel));
  squared_sum_image.reset(new GpuMat<float>(width, height, channel));
}

void GpuMatRefImage::Filter(const float* image_data,
                            const size_t window_radius,
                            const size_t window_step, const float sigma_spatial,
                            const float sigma_color) {
  // adding channel as input
  CudaArrayWrapper<float> image_array(width_, height_, channel_);
  image_array.CopyToDevice(image_data);
  image_texture.addressMode[0] = cudaAddressModeBorder;
  image_texture.addressMode[1] = cudaAddressModeBorder;
  image_texture.addressMode[2] = cudaAddressModeBorder;
  image_texture.filterMode = cudaFilterModePoint;
  image_texture.normalized = false;

  const dim3 block_size(kBlockDimX, kBlockDimY, kBlockDimZ);
  const dim3 grid_size((width_ - 1) / block_size.x + 1,
                       (height_ - 1) / block_size.y + 1,
                       (channel_ - 1) / block_size.z + 1);

  CUDA_SAFE_CALL(cudaBindTextureToArray(image_texture, image_array.GetPtr()));
  FilterKernel<<<grid_size, block_size>>>(
      *image, *sum_image, *squared_sum_image, window_radius, window_step,
      sigma_spatial, sigma_color);
  CUDA_SYNC_AND_CHECK();
  CUDA_SAFE_CALL(cudaUnbindTexture(image_texture));
}

}  // namespace mvs
}  // namespace colmap
