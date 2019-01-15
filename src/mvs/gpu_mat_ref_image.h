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

#ifndef COLMAP_SRC_MVS_GPU_MAT_REF_IMAGE_H_
#define COLMAP_SRC_MVS_GPU_MAT_REF_IMAGE_H_

#include <memory>

#include "mvs/cuda_array_wrapper.h"
#include "mvs/gpu_mat.h"

namespace colmap {
namespace mvs {

class GpuMatRefImage {
 public:
  GpuMatRefImage(const size_t width, const size_t height, const size_t channel);

  // Filter image using sum convolution kernel to compute local sum of
  // intensities. The filtered images can then be used for repeated, efficient
  // NCC computation.
  void Filter(const float * image_data, const size_t window_radius,
              const size_t window_step, const float sigma_spatial,
              const float sigma_color);

  // Image features.
  std::unique_ptr<GpuMat<float>> image;

  // Local sum of image intensities.
  std::unique_ptr<GpuMat<float>> sum_image;

  // Local sum of squared image intensities.
  std::unique_ptr<GpuMat<float>> squared_sum_image;

 private:
  const static size_t kBlockDimX = 16;
  const static size_t kBlockDimY = 12;

  size_t width_;
  size_t height_;
  size_t channel_;
};



/**
 * Computes
 */
struct MultiChannelWeightComputer {
  __device__ MultiChannelWeightComputer(const float sigma_spatial,
                                        const float sigma_feature)
      : spatial_normalization_(1.0f / (2.0f * sigma_spatial * sigma_spatial)),
        feature_normalization_(1.0f / (2.0f * sigma_feature* sigma_feature)) {}

  __device__ inline float Compute(const float row_diff, const float col_diff,
                                  const float * ref_feature,
                                  const float * src_feature, 
                                  const int num_channels) const {

    const float spatial_dist_squared =
        row_diff * row_diff + col_diff * col_diff;

    float feature_dist_sq = 0.0f;
    for(auto i = 0; i < num_channels; i++){
      auto diff = ref_feature[i] + src_feature[i];
      feature_dist_sq += diff*diff;
    }

    return exp(-spatial_dist_squared * spatial_normalization_ -
               feature_dist_sq * feature_normalization_);
  }

 private:
  const float spatial_normalization_;
  const float feature_normalization_;
};

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_GPU_MAT_REF_IMAGE_H_
