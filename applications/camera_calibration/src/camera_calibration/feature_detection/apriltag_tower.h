// Copyright 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <memory>

#include "camera_calibration/feature_detection/feature_detector.h"

namespace vis {

struct AprilTagTowerDetectorPrivate;

/// Feature detector for a metric 3D AprilTag tower.
///
/// Feature ids are tag_id * 4 + corner_id. Corner ids follow the AprilTag
/// detector's counter-clockwise quad order around the physical tag border.
class AprilTagTowerDetector : public FeatureDetector {
 public:
  AprilTagTowerDetector(
      const vector<string>& pattern_yaml_paths,
      int window_half_extent);

  ~AprilTagTowerDetector();

  virtual bool SetPatternYAMLPaths(
      const vector<string>& paths) override;

  virtual void DetectFeatures(
      const Image<Vec3u8>& image,
      vector<PointFeature>* features,
      Image<Vec3u8>* detection_visualization) override;

  virtual int GetPatternCount() const override;

  virtual void GetCorners(
      int pattern_index,
      unordered_map<int, Vec2i>* feature_id_to_coord) const override;

  virtual void GetKnownGeometries(vector<KnownGeometry>* known_geometries) const override;

  inline bool valid() const { return valid_; }

 private:
  unique_ptr<AprilTagTowerDetectorPrivate> d;
  bool valid_;
};

}
