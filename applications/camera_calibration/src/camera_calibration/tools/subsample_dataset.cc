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

#include "camera_calibration/tools/tools.h"

#include <algorithm>

#include <libvis/logging.h>

#include "camera_calibration/dataset.h"
#include "camera_calibration/io/calibration_io.h"

namespace vis {
namespace {

int CountFeatures(const Dataset& dataset) {
  int count = 0;
  for (int imageset_index = 0; imageset_index < dataset.ImagesetCount(); ++ imageset_index) {
    shared_ptr<const Imageset> imageset = dataset.GetImageset(imageset_index);
    for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
      count += imageset->FeaturesOfCamera(camera_index).size();
    }
  }
  return count;
}

bool LookupPatternGridPosition(
    const Dataset& dataset,
    int feature_id,
    Vec2i* position) {
  for (int geometry_index = 0; geometry_index < dataset.KnownGeometriesCount(); ++ geometry_index) {
    const KnownGeometry& geometry = dataset.GetKnownGeometry(geometry_index);
    auto it = geometry.feature_id_to_position.find(feature_id);
    if (it != geometry.feature_id_to_position.end()) {
      *position = it->second;
      return true;
    }
  }
  return false;
}

bool KeepFeature(
    const Dataset& dataset,
    const PointFeature& feature,
    int pattern_grid_stride,
    int id_stride) {
  if (pattern_grid_stride > 1) {
    Vec2i position;
    if (LookupPatternGridPosition(dataset, feature.id, &position)) {
      return position.x() % pattern_grid_stride == 0 &&
             position.y() % pattern_grid_stride == 0;
    }
  }

  if (id_stride > 1) {
    int id_mod = feature.id % id_stride;
    if (id_mod < 0) {
      id_mod += id_stride;
    }
    return id_mod == 0;
  }

  return true;
}

void CopyKnownGeometries(const Dataset& input, Dataset* output) {
  output->SetKnownGeometriesCount(input.KnownGeometriesCount());
  for (int geometry_index = 0; geometry_index < input.KnownGeometriesCount(); ++ geometry_index) {
    output->GetKnownGeometry(geometry_index) = input.GetKnownGeometry(geometry_index);
  }
}

}  // namespace

int SubsampleDataset(
    const string& dataset_path,
    const string& output_path,
    int frame_stride,
    int pattern_grid_stride,
    int id_stride,
    int min_features_per_camera_view) {
  if (dataset_path.empty() || output_path.empty()) {
    LOG(ERROR) << "--subsample_dataset requires one --dataset_files entry and --dataset_output_path.";
    return EXIT_FAILURE;
  }
  if (frame_stride < 1 || pattern_grid_stride < 1 || id_stride < 1 ||
      min_features_per_camera_view < 0) {
    LOG(ERROR) << "Invalid dataset subsampling parameters.";
    return EXIT_FAILURE;
  }

  Dataset input(1);
  if (!LoadDataset(dataset_path.c_str(), &input)) {
    return EXIT_FAILURE;
  }

  Dataset output(input.num_cameras());
  for (int camera_index = 0; camera_index < input.num_cameras(); ++ camera_index) {
    output.SetImageSize(camera_index, input.GetImageSize(camera_index));
  }
  CopyKnownGeometries(input, &output);

  vector<int> per_camera_counts(input.num_cameras(), 0);
  int copied_imagesets = 0;
  int fallback_view_count = 0;

  for (int imageset_index = 0; imageset_index < input.ImagesetCount(); ++ imageset_index) {
    if (imageset_index % frame_stride != 0) {
      continue;
    }

    shared_ptr<const Imageset> input_imageset = input.GetImageset(imageset_index);
    shared_ptr<Imageset> output_imageset = output.NewImageset();
    output_imageset->SetFilename(input_imageset->GetFilename());
    bool non_empty = false;

    for (int camera_index = 0; camera_index < input.num_cameras(); ++ camera_index) {
      const vector<PointFeature>& input_features =
          input_imageset->FeaturesOfCamera(camera_index);
      vector<PointFeature>& output_features =
          output_imageset->FeaturesOfCamera(camera_index);
      output_features.reserve(input_features.size());

      for (const PointFeature& feature : input_features) {
        if (KeepFeature(input, feature, pattern_grid_stride, id_stride)) {
          output_features.push_back(feature);
        }
      }

      if (!input_features.empty() &&
          static_cast<int>(output_features.size()) < min_features_per_camera_view) {
        output_features = input_features;
        ++ fallback_view_count;
      }

      per_camera_counts[camera_index] += output_features.size();
      non_empty |= !output_features.empty();
    }

    if (!non_empty) {
      output.DeleteLastImageset();
      continue;
    }
    ++ copied_imagesets;
  }

  if (!SaveDataset(output_path.c_str(), output)) {
    LOG(ERROR) << "Could not save subsampled dataset to: " << output_path;
    return EXIT_FAILURE;
  }

  const int input_feature_count = CountFeatures(input);
  const int output_feature_count = CountFeatures(output);
  LOG(INFO) << "Subsampled " << dataset_path << " -> " << output_path;
  LOG(INFO) << "Imagesets: " << input.ImagesetCount() << " -> " << copied_imagesets;
  LOG(INFO) << "Features: " << input_feature_count << " -> " << output_feature_count;
  LOG(INFO) << "Views kept dense because of min_features_per_camera_view: "
            << fallback_view_count;
  for (int camera_index = 0; camera_index < output.num_cameras(); ++ camera_index) {
    LOG(INFO) << "Camera " << camera_index << " retained features: "
              << per_camera_counts[camera_index];
  }

  return EXIT_SUCCESS;
}

}  // namespace vis
