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

#ifndef COLMAP_SRC_UTIL_CNN_FEATURE_H_
#define COLMAP_SRC_UTIL_CNN_FEATURE_H_

#include <algorithm>
#include <cmath>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif
#include <FreeImage.h>

#include "cnpy/cnpy.h"
#include "util/string.h"
#include <iostream>

namespace colmap {
// Wrapper class around FreeImage CNNFeatures.
class CNNFeature {
 public:
  CNNFeature();

  // Copy constructor.
  CNNFeature(const CNNFeature& other);
  // Move constructor.
  CNNFeature(CNNFeature&& other);

  // Create CNNFeature object from existing FreeImage CNNFeature object. Note that
  // this class takes ownership of the object.
  explicit CNNFeature(cnpy::NpyArray * data);

  // Copy assignment.
  CNNFeature& operator=(const CNNFeature& other);
  // Move assignment.
  CNNFeature& operator=(CNNFeature&& other);

  // Dimensions of CNNFeature.
  int Width() const;
  int Height() const;
  int Channels() const;
  float * Data();
  const float * Data() const;

  // Number of bytes required to store image.
  size_t NumBytes() const;

  // Copy raw image data to array.
  std::vector<float> ConvertToRowMajorArray() const;

  bool Read(const std::string& path);
  bool Write(const std::string& path) const;

  // Rescale image to the new dimensions.
  void Rescale(const int new_width, const int new_height);

 private:

  std::shared_ptr<cnpy::NpyArray> data_ = nullptr;
  int width_;
  int height_;
  int channels_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_CNN_FEATURE_H_
