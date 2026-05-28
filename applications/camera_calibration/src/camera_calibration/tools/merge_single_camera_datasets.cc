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

#include <cmath>
#include <map>

#include <libvis/logging.h>

#include "camera_calibration/dataset.h"
#include "camera_calibration/io/calibration_io.h"

namespace vis {
namespace {

bool KnownGeometryIsCompatible(
    const KnownGeometry& a,
    const KnownGeometry& b) {
  if (std::fabs(a.cell_length_in_meters - b.cell_length_in_meters) > 1e-8f) {
    return false;
  }
  if (a.feature_id_to_position.size() != b.feature_id_to_position.size() ||
      a.feature_id_to_position3d.size() != b.feature_id_to_position3d.size()) {
    return false;
  }
  return true;
}

int CountFeatures(const Dataset& dataset) {
  int count = 0;
  for (int imageset_index = 0; imageset_index < dataset.ImagesetCount(); ++ imageset_index) {
    for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
      count += dataset.GetImageset(imageset_index)->FeaturesOfCamera(camera_index).size();
    }
  }
  return count;
}

}  // namespace

int MergeSingleCameraDatasets(
    const vector<string>& dataset_files,
    const string& output_path) {
  if (dataset_files.empty() || output_path.empty()) {
    LOG(ERROR) << "--merge_single_camera_datasets requires --dataset_files and --dataset_output_path.";
    return EXIT_FAILURE;
  }

  vector<Dataset> shards;
  shards.reserve(dataset_files.size());

  for (const string& path : dataset_files) {
    shards.emplace_back(1);
    Dataset& shard = shards.back();
    if (!LoadDataset(path.c_str(), &shard)) {
      return EXIT_FAILURE;
    }
    if (shard.num_cameras() != 1) {
      LOG(ERROR) << "Expected a single-camera dataset shard, got "
                 << shard.num_cameras() << " cameras in: " << path;
      return EXIT_FAILURE;
    }
  }

  const int camera_count = shards.size();
  Dataset merged(camera_count);
  for (int camera_index = 0; camera_index < camera_count; ++ camera_index) {
    merged.SetImageSize(camera_index, shards[camera_index].GetImageSize(0));
  }

  const int known_geometry_count = shards[0].KnownGeometriesCount();
  merged.SetKnownGeometriesCount(known_geometry_count);
  for (int geometry_index = 0; geometry_index < known_geometry_count; ++ geometry_index) {
    merged.GetKnownGeometry(geometry_index) = shards[0].GetKnownGeometry(geometry_index);
  }

  for (int camera_index = 1; camera_index < camera_count; ++ camera_index) {
    if (shards[camera_index].KnownGeometriesCount() != known_geometry_count) {
      LOG(ERROR) << "Known geometry count mismatch in shard " << camera_index
                 << ": " << dataset_files[camera_index];
      return EXIT_FAILURE;
    }
    for (int geometry_index = 0; geometry_index < known_geometry_count; ++ geometry_index) {
      if (!KnownGeometryIsCompatible(
              shards[0].GetKnownGeometry(geometry_index),
              shards[camera_index].GetKnownGeometry(geometry_index))) {
        LOG(ERROR) << "Known geometry mismatch in shard " << camera_index
                   << ": " << dataset_files[camera_index];
        return EXIT_FAILURE;
      }
    }
  }

  vector<map<string, vector<PointFeature>>> features_by_camera(camera_count);
  map<string, bool> filename_union;
  for (int camera_index = 0; camera_index < camera_count; ++ camera_index) {
    const Dataset& shard = shards[camera_index];
    for (int imageset_index = 0; imageset_index < shard.ImagesetCount(); ++ imageset_index) {
      shared_ptr<const Imageset> imageset = shard.GetImageset(imageset_index);
      const string& filename = imageset->GetFilename();
      filename_union[filename] = true;
      features_by_camera[camera_index][filename] =
          imageset->FeaturesOfCamera(0);
    }
  }

  int merged_feature_count = 0;
  for (const auto& filename_item : filename_union) {
    shared_ptr<Imageset> imageset = merged.NewImageset();
    imageset->SetFilename(filename_item.first);

    bool non_empty = false;
    for (int camera_index = 0; camera_index < camera_count; ++ camera_index) {
      auto features_it = features_by_camera[camera_index].find(filename_item.first);
      if (features_it == features_by_camera[camera_index].end()) {
        continue;
      }
      vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);
      features = features_it->second;
      merged_feature_count += features.size();
      non_empty |= !features.empty();
    }

    if (!non_empty) {
      merged.DeleteLastImageset();
    }
  }

  if (!SaveDataset(output_path.c_str(), merged)) {
    LOG(ERROR) << "Could not save merged dataset to: " << output_path;
    return EXIT_FAILURE;
  }

  int shard_feature_count = 0;
  for (const Dataset& shard : shards) {
    shard_feature_count += CountFeatures(shard);
  }
  LOG(INFO) << "Merged " << camera_count << " single-camera shards into "
            << output_path << " with " << merged.ImagesetCount()
            << " imagesets and " << merged_feature_count << " features.";
  if (merged_feature_count != shard_feature_count) {
    LOG(WARNING) << "Merged feature count (" << merged_feature_count
                 << ") differs from shard feature count (" << shard_feature_count << ").";
  }

  return EXIT_SUCCESS;
}

}  // namespace vis
