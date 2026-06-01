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

#include "camera_calibration/feature_detection/apriltag_tower.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

#include <apriltag.h>
#include <libvis/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <tag36h11.h>
#include <yaml-cpp/yaml.h>

namespace vis {

struct AprilTagTowerConfig {
  string tag_family = "tag36h11";
  int faces = 8;
  int tag_columns = 2;
  int tag_rows = 16;
  float tag_size_m = 0.08f;
  float tag_spacing_m = 0.02f;
  int first_tag_id = 0;
  int face_id_stride = 32;
  float face_width_m = 0.18f;
  float face0_angle_degrees = 0;
  int tag_rotation_degrees = 0;
  int max_hamming = 2;
  int detector_threads = 4;
  bool corner_subpixel_refinement = true;
  int corner_subpixel_window_half_extent = 5;
  int corner_subpixel_max_iterations = 30;
  float corner_subpixel_epsilon = 0.01f;
  float corner_subpixel_max_shift_px = 2.0f;
};

struct AprilTagTowerDetectorPrivate {
  ~AprilTagTowerDetectorPrivate() {
    if (apriltag_detector) {
      apriltag_detector_destroy(apriltag_detector);
    }
    if (apriltag_family) {
      tag36h11_destroy(apriltag_family);
    }
  }

  void PrepareAprilTagDetector() {
    if (!apriltag_detector) {
      apriltag_detector = apriltag_detector_create();
      apriltag_family = tag36h11_create();
      apriltag_detector_add_family_bits(
          apriltag_detector,
          apriltag_family,
          config.max_hamming);

      apriltag_detector->quad_decimate = 1.0;
      apriltag_detector->quad_sigma = 0.0;
      apriltag_detector->refine_edges = 1;
      apriltag_detector->decode_sharpening = 0.25;
      apriltag_detector->nthreads = std::max(1, config.detector_threads);
    }
  }

  AprilTagTowerConfig config;
  vector<KnownGeometry> known_geometries;

  apriltag_detector_t* apriltag_detector = nullptr;
  apriltag_family_t* apriltag_family = nullptr;
};

namespace {

template <typename T>
T ReadOptional(const YAML::Node& node, const char* key, const T& default_value) {
  if (node[key]) {
    return node[key].as<T>();
  }
  return default_value;
}

bool LoadTowerConfig(const string& path, AprilTagTowerConfig* config) {
  try {
    YAML::Node node = YAML::LoadFile(path);
    if (node.IsNull()) {
      LOG(ERROR) << "Cannot read file: " << path;
      return false;
    }

    string type = ReadOptional<string>(node, "type", "apriltag_tower");
    if (type != "apriltag_tower") {
      LOG(ERROR) << "Unsupported AprilTag tower config type in " << path << ": " << type;
      return false;
    }

    config->tag_family = ReadOptional<string>(node, "tag_family", config->tag_family);
    config->faces = ReadOptional<int>(node, "faces", config->faces);
    config->tag_columns = ReadOptional<int>(node, "tag_columns", config->tag_columns);
    config->tag_rows = ReadOptional<int>(node, "tag_rows", config->tag_rows);
    config->tag_size_m = ReadOptional<float>(node, "tag_size_m", config->tag_size_m);
    config->tag_spacing_m = ReadOptional<float>(node, "tag_spacing_m", config->tag_spacing_m);
    config->first_tag_id = ReadOptional<int>(node, "first_tag_id", config->first_tag_id);
    config->face_id_stride = ReadOptional<int>(
        node,
        "face_id_stride",
        config->tag_columns * config->tag_rows);
    config->face_width_m = ReadOptional<float>(
        node,
        "face_width_m",
        config->tag_columns * config->tag_size_m + (config->tag_columns - 1) * config->tag_spacing_m);
    config->face0_angle_degrees = ReadOptional<float>(node, "face0_angle_degrees", config->face0_angle_degrees);
    config->tag_rotation_degrees = ReadOptional<int>(node, "tag_rotation_degrees", config->tag_rotation_degrees);
    config->max_hamming = ReadOptional<int>(node, "max_hamming", config->max_hamming);
    config->detector_threads = ReadOptional<int>(
        node,
        "detector_threads",
        static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 2)));
    config->corner_subpixel_refinement = ReadOptional<bool>(
        node,
        "corner_subpixel_refinement",
        config->corner_subpixel_refinement);
    config->corner_subpixel_window_half_extent = ReadOptional<int>(
        node,
        "corner_subpixel_window_half_extent",
        config->corner_subpixel_window_half_extent);
    config->corner_subpixel_max_iterations = ReadOptional<int>(
        node,
        "corner_subpixel_max_iterations",
        config->corner_subpixel_max_iterations);
    config->corner_subpixel_epsilon = ReadOptional<float>(
        node,
        "corner_subpixel_epsilon",
        config->corner_subpixel_epsilon);
    config->corner_subpixel_max_shift_px = ReadOptional<float>(
        node,
        "corner_subpixel_max_shift_px",
        config->corner_subpixel_max_shift_px);
  } catch (const YAML::Exception& ex) {
    LOG(ERROR) << "Cannot parse AprilTag tower config: " << path << " (" << ex.what() << ")";
    return false;
  }

  if (config->tag_family != "tag36h11") {
    LOG(ERROR) << "Only tag36h11 is supported for AprilTag tower detection at the moment.";
    return false;
  }
  if (config->faces <= 2 ||
      config->tag_columns <= 0 ||
      config->tag_rows <= 0 ||
      config->tag_size_m <= 0 ||
      config->tag_spacing_m < 0 ||
      config->face_id_stride < config->tag_columns * config->tag_rows ||
      config->face_width_m <= 0 ||
      (config->tag_rotation_degrees != 0 && config->tag_rotation_degrees != 180) ||
      config->max_hamming < 0 ||
      config->max_hamming > 2 ||
      config->detector_threads <= 0 ||
      config->corner_subpixel_window_half_extent <= 0 ||
      config->corner_subpixel_max_iterations <= 0 ||
      config->corner_subpixel_epsilon <= 0 ||
      config->corner_subpixel_max_shift_px < 0) {
    LOG(ERROR) << "Invalid AprilTag tower dimensions in: " << path;
    return false;
  }

  return true;
}

int PhysicalCornerForTagRotation(int corner, int tag_rotation_degrees) {
  if (tag_rotation_degrees == 180) {
    return (corner + 2) % 4;
  }
  return corner;
}

KnownGeometry BuildTowerGeometry(const AprilTagTowerConfig& config) {
  KnownGeometry geometry;
  geometry.cell_length_in_meters = config.tag_size_m + config.tag_spacing_m;

  const float pitch = config.tag_size_m + config.tag_spacing_m;
  const float half_tag = 0.5f * config.tag_size_m;
  const float apothem = config.face_width_m / (2.0f * std::tan(static_cast<float>(M_PI) / config.faces));
  const float face0_angle = static_cast<float>(M_PI / 180.0) * config.face0_angle_degrees;

  for (int face = 0; face < config.faces; ++ face) {
    const float theta = face0_angle + face * 2.0f * static_cast<float>(M_PI) / config.faces;
    const Vec3f normal(std::cos(theta), std::sin(theta), 0);
    const Vec3f u_axis(-std::sin(theta), std::cos(theta), 0);
    const Vec3f z_axis(0, 0, 1);

    for (int row = 0; row < config.tag_rows; ++ row) {
      for (int col = 0; col < config.tag_columns; ++ col) {
        const int local_tag_id = row * config.tag_columns + col;
        const int tag_id = config.first_tag_id + face * config.face_id_stride + local_tag_id;
        const float center_u = (col - 0.5f * (config.tag_columns - 1)) * pitch;
        const float center_z = (row - 0.5f * (config.tag_rows - 1)) * pitch;
        const Vec3f center = normal * apothem + u_axis * center_u + z_axis * center_z;

        const Vec3f physical_corners[4] = {
            center - u_axis * half_tag - z_axis * half_tag,
            center + u_axis * half_tag - z_axis * half_tag,
            center + u_axis * half_tag + z_axis * half_tag,
            center - u_axis * half_tag + z_axis * half_tag};
        for (int corner = 0; corner < 4; ++ corner) {
          const int physical_corner = PhysicalCornerForTagRotation(
              corner,
              config.tag_rotation_degrees);
          geometry.feature_id_to_position3d[tag_id * 4 + corner] =
              physical_corners[physical_corner];
        }
      }
    }
  }

  return geometry;
}

int ClampToImage(int value, int max_value) {
  return std::max(0, std::min(value, max_value));
}

bool IsFinitePoint(const cv::Point2f& point) {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool CanRefineCorner(const cv::Point2f& point, const Image<u8>& gray_image, int window_half_extent) {
  return point.x >= window_half_extent + 1 &&
         point.y >= window_half_extent + 1 &&
         point.x < static_cast<float>(gray_image.width() - window_half_extent - 1) &&
         point.y < static_cast<float>(gray_image.height() - window_half_extent - 1);
}

void RefineTagCornersSubpixel(
    const Image<u8>& gray_image,
    const AprilTagTowerConfig& config,
    const apriltag_detection_t& tag_detection,
    Vec2f refined_corners[4]) {
  for (int corner = 0; corner < 4; ++ corner) {
    refined_corners[corner] = Vec2f(
        tag_detection.p[corner][0],
        tag_detection.p[corner][1]);
  }

  if (!config.corner_subpixel_refinement) {
    return;
  }

  const int window_half_extent = config.corner_subpixel_window_half_extent;
  vector<int> corner_indices;
  vector<cv::Point2f> points;
  corner_indices.reserve(4);
  points.reserve(4);
  for (int corner = 0; corner < 4; ++ corner) {
    const cv::Point2f point(
        tag_detection.p[corner][0],
        tag_detection.p[corner][1]);
    if (CanRefineCorner(point, gray_image, window_half_extent)) {
      corner_indices.push_back(corner);
      points.push_back(point);
    }
  }

  if (points.empty()) {
    return;
  }

  cv::Mat gray_mat(
      static_cast<int>(gray_image.height()),
      static_cast<int>(gray_image.width()),
      CV_8UC1,
      const_cast<u8*>(gray_image.data()),
      static_cast<size_t>(gray_image.stride()));
  cv::cornerSubPix(
      gray_mat,
      points,
      cv::Size(window_half_extent, window_half_extent),
      cv::Size(-1, -1),
      cv::TermCriteria(
          cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
          config.corner_subpixel_max_iterations,
          config.corner_subpixel_epsilon));

  const float max_shift_sq =
      config.corner_subpixel_max_shift_px * config.corner_subpixel_max_shift_px;
  for (size_t i = 0; i < points.size(); ++ i) {
    const int corner = corner_indices[i];
    const cv::Point2f original(
        tag_detection.p[corner][0],
        tag_detection.p[corner][1]);
    const cv::Point2f refined = points[i];
    const float dx = refined.x - original.x;
    const float dy = refined.y - original.y;
    if (IsFinitePoint(refined) &&
        (max_shift_sq == 0 || dx * dx + dy * dy <= max_shift_sq)) {
      refined_corners[corner] = Vec2f(refined.x, refined.y);
    }
  }
}

}  // namespace

AprilTagTowerDetector::AprilTagTowerDetector(
    const vector<string>& pattern_yaml_paths,
    int window_half_extent) {
  valid_ = true;
  d.reset(new AprilTagTowerDetectorPrivate());
  this->window_half_extent = window_half_extent;
  this->refinement_type = FeatureRefinement::NoRefinement;
  this->cell_length_in_meters = 0.1f;

  SetPatternYAMLPaths(pattern_yaml_paths);
}

AprilTagTowerDetector::~AprilTagTowerDetector() {
  // required for unique_ptr with type that is incomplete in the header
}

bool AprilTagTowerDetector::SetPatternYAMLPaths(
    const vector<string>& paths) {
  pattern_yaml_paths = paths;
  d->known_geometries.clear();
  valid_ = true;

  if (paths.empty()) {
    return true;
  }
  if (paths.size() != 1) {
    LOG(ERROR) << "AprilTagTowerDetector expects exactly one tower config.";
    valid_ = false;
    return false;
  }

  if (!LoadTowerConfig(paths[0], &d->config)) {
    valid_ = false;
    return false;
  }

  cell_length_in_meters = d->config.tag_size_m + d->config.tag_spacing_m;
  d->known_geometries.push_back(BuildTowerGeometry(d->config));
  return true;
}

void AprilTagTowerDetector::DetectFeatures(
    const Image<Vec3u8>& image,
    vector<PointFeature>* features,
    Image<Vec3u8>* detection_visualization) {
  features->clear();

  if (detection_visualization) {
    detection_visualization->SetSize(image.size());
    detection_visualization->SetTo(image);
  }

  if (!valid_ || d->known_geometries.empty()) {
    return;
  }

  d->PrepareAprilTagDetector();

  Image<u8> gray_image;
  image.ConvertToGrayscale(&gray_image);

  image_u8_t apriltag_image = {
      static_cast<int32_t>(gray_image.width()),
      static_cast<int32_t>(gray_image.height()),
      static_cast<int32_t>(gray_image.stride()),
      const_cast<u8*>(gray_image.data())};

  zarray_t* detections = apriltag_detector_detect(d->apriltag_detector, &apriltag_image);
  const KnownGeometry& geometry = d->known_geometries[0];

  for (int detection_index = 0; detection_index < zarray_size(detections); ++ detection_index) {
    apriltag_detection_t* tag_detection;
    zarray_get(detections, detection_index, &tag_detection);

    if (tag_detection->hamming > d->config.max_hamming) {
      continue;
    }

    const int first_feature_id = tag_detection->id * 4;
    if (!geometry.HasFeature(first_feature_id)) {
      continue;
    }

    Vec2f refined_corners[4];
    RefineTagCornersSubpixel(gray_image, d->config, *tag_detection, refined_corners);

    for (int corner = 0; corner < 4; ++ corner) {
      features->emplace_back(
          refined_corners[corner],
          first_feature_id + corner);
    }

    if (detection_visualization) {
      for (int corner = 0; corner < 4; ++ corner) {
        int next_corner = (corner + 1) % 4;
        int x0 = ClampToImage(static_cast<int>(refined_corners[corner].x() + 0.5), image.width() - 1);
        int y0 = ClampToImage(static_cast<int>(refined_corners[corner].y() + 0.5), image.height() - 1);
        int x1 = ClampToImage(static_cast<int>(refined_corners[next_corner].x() + 0.5), image.width() - 1);
        int y1 = ClampToImage(static_cast<int>(refined_corners[next_corner].y() + 0.5), image.height() - 1);
        detection_visualization->DrawLine(x0, y0, x1, y1, Vec3u8(0, 255, 0));
      }
    }
  }

  apriltag_detections_destroy(detections);
}

int AprilTagTowerDetector::GetPatternCount() const {
  return d->known_geometries.size();
}

void AprilTagTowerDetector::GetCorners(
    int /*pattern_index*/,
    unordered_map<int, Vec2i>* feature_id_to_coord) const {
  feature_id_to_coord->clear();
}

void AprilTagTowerDetector::GetKnownGeometries(vector<KnownGeometry>* known_geometries) const {
  *known_geometries = d->known_geometries;
}

}  // namespace vis
