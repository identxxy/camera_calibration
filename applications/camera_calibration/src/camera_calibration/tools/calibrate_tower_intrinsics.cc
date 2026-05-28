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
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "camera_calibration/dataset.h"
#include "camera_calibration/io/calibration_io.h"
#include "camera_calibration/models/central_opencv.h"

namespace vis {
namespace {

struct CameraManifestEntry {
  string stage_name;
  string machine;
  string user_id;
  string source_dir;
};

struct CameraStats {
  int total_views = 0;
  int positive_views = 0;
  int usable_views = 0;
  int total_points = 0;
  int usable_points = 0;
  int max_points_per_view = 0;
  double bbox_area_ratio = 0;
  double rms = std::numeric_limits<double>::quiet_NaN();
  string status;
  string reason;
};

vector<string> SplitTabs(const string& line) {
  vector<string> result;
  std::stringstream stream(line);
  string item;
  while (std::getline(stream, item, '\t')) {
    result.push_back(item);
  }
  return result;
}

unordered_map<int, CameraManifestEntry> LoadCameraManifest(const string& path) {
  unordered_map<int, CameraManifestEntry> entries;
  if (path.empty()) {
    return entries;
  }

  std::ifstream stream(path);
  if (!stream) {
    LOG(WARNING) << "Could not read camera manifest: " << path;
    return entries;
  }

  string line;
  if (!std::getline(stream, line)) {
    return entries;
  }

  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    vector<string> fields = SplitTabs(line);
    if (fields.size() < 5) {
      continue;
    }

    CameraManifestEntry entry;
    int camera_index = std::stoi(fields[0]);
    entry.stage_name = fields[1];
    entry.machine = fields[2];
    entry.user_id = fields[3];
    entry.source_dir = fields[4];
    entries[camera_index] = entry;
  }

  return entries;
}

CameraManifestEntry DefaultManifestEntry(int camera_index) {
  CameraManifestEntry entry;
  std::stringstream name;
  name << "camera_" << std::setw(2) << std::setfill('0') << camera_index;
  entry.stage_name = name.str();
  entry.machine = "";
  entry.user_id = std::to_string(camera_index);
  entry.source_dir = "";
  return entry;
}

string FormatDouble(double value) {
  if (!std::isfinite(value)) {
    return "";
  }
  std::ostringstream stream;
  stream << std::setprecision(14) << value;
  return stream.str();
}

void WriteOpenCVYaml(
    const string& path,
    int width,
    int height,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    double rms) {
  cv::FileStorage storage(path, cv::FileStorage::WRITE);
  storage << "image_width" << width;
  storage << "image_height" << height;
  storage << "camera_matrix" << camera_matrix;
  storage << "distortion_model" << "opencv_radtan";
  storage << "distortion_coefficients" << dist_coeffs;
  storage << "rms_reprojection_error" << rms;
}

void FillCentralOpenCVModel(
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    CentralOpenCVModel* model) {
  model->parameters().setZero();
  model->parameters()(0) = camera_matrix.at<double>(0, 0);
  model->parameters()(1) = camera_matrix.at<double>(1, 1);
  model->parameters()(2) = camera_matrix.at<double>(0, 2);
  model->parameters()(3) = camera_matrix.at<double>(1, 2);

  const int coeff_count = dist_coeffs.total();
  auto coeff = [&](int index) -> double {
    return index < coeff_count ? dist_coeffs.at<double>(index, 0) : 0.0;
  };

  // OpenCV default order: k1, k2, p1, p2, k3. The repository model order is
  // fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2.
  model->parameters()(4) = coeff(0);
  model->parameters()(5) = coeff(1);
  model->parameters()(6) = coeff(4);
  model->parameters()(7) = 0;
  model->parameters()(8) = 0;
  model->parameters()(9) = 0;
  model->parameters()(10) = coeff(2);
  model->parameters()(11) = coeff(3);
}

}  // namespace

int CalibrateTowerIntrinsics(
    const string& dataset_path,
    const string& camera_manifest_path,
    const string& output_directory,
    int min_points_per_view,
    int min_views,
    int corner_id_offset) {
  if (dataset_path.empty() || output_directory.empty()) {
    LOG(ERROR) << "--calibrate_tower_intrinsics requires --dataset_files and --output_directory.";
    return EXIT_FAILURE;
  }
  if (min_points_per_view < 4) {
    LOG(ERROR) << "--tower_intrinsics_min_points_per_view must be at least 4.";
    return EXIT_FAILURE;
  }
  if (min_views < 1) {
    LOG(ERROR) << "--tower_intrinsics_min_views must be positive.";
    return EXIT_FAILURE;
  }
  if (corner_id_offset < 0 || corner_id_offset > 3) {
    LOG(ERROR) << "--tower_intrinsics_corner_id_offset must be in [0, 3].";
    return EXIT_FAILURE;
  }

  Dataset dataset(0);
  if (!LoadDataset(dataset_path.c_str(), &dataset)) {
    return EXIT_FAILURE;
  }
  if (dataset.KnownGeometriesCount() == 0) {
    LOG(ERROR) << "The dataset does not contain known pattern geometry.";
    return EXIT_FAILURE;
  }

  boost::filesystem::create_directories(output_directory);
  const unordered_map<int, CameraManifestEntry> manifest =
      LoadCameraManifest(camera_manifest_path);

  const KnownGeometry& geometry = dataset.GetKnownGeometry(0);
  const boost::filesystem::path output_path(output_directory);
  std::ofstream summary((output_path / "intrinsics_summary.tsv").string());
  std::ofstream insufficient((output_path / "insufficient_cameras.tsv").string());

  summary << "camera_index\tstage_name\tmachine\tuser_id\tstatus\treason\twidth\theight"
          << "\ttotal_views\tpositive_views\tusable_views\ttotal_points\tusable_points"
          << "\tmax_points_per_view\tbbox_area_ratio\trms"
          << "\tfx\tfy\tcx\tcy\tk1\tk2\tp1\tp2\tk3\n";
  insufficient << "camera_index\tstage_name\tmachine\tuser_id\treason"
               << "\tpositive_views\tusable_views\ttotal_points\tusable_points"
               << "\tmax_points_per_view\tbbox_area_ratio\n";

  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    CameraManifestEntry entry = DefaultManifestEntry(camera_index);
    auto manifest_it = manifest.find(camera_index);
    if (manifest_it != manifest.end()) {
      entry = manifest_it->second;
    }

    const Vec2i image_size = dataset.GetImageSize(camera_index);
    vector<vector<cv::Point3f>> object_points;
    vector<vector<cv::Point2f>> image_points;
    vector<vector<int>> feature_ids;
    vector<int> usable_imageset_indices;
    CameraStats stats;
    stats.total_views = dataset.ImagesetCount();

    float min_x = std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();

    for (int imageset_index = 0; imageset_index < dataset.ImagesetCount(); ++ imageset_index) {
      const vector<PointFeature>& features =
          dataset.GetImageset(imageset_index)->FeaturesOfCamera(camera_index);
      if (!features.empty()) {
        ++ stats.positive_views;
      }
      stats.total_points += features.size();
      stats.max_points_per_view = std::max<int>(stats.max_points_per_view, features.size());

      vector<cv::Point3f> frame_object_points;
      vector<cv::Point2f> frame_image_points;
      vector<int> frame_feature_ids;
      for (const PointFeature& feature : features) {
        Vec3f point;
        const int tag_id = feature.id / 4;
        const int corner_id = feature.id % 4;
        const int geometry_feature_id =
            tag_id * 4 + ((corner_id + corner_id_offset) % 4);
        if (!geometry.GetFeaturePoint3D(geometry_feature_id, &point)) {
          continue;
        }
        frame_object_points.emplace_back(point.x(), point.y(), point.z());
        frame_image_points.emplace_back(feature.xy.x(), feature.xy.y());
        frame_feature_ids.push_back(geometry_feature_id);
        min_x = std::min(min_x, feature.xy.x());
        min_y = std::min(min_y, feature.xy.y());
        max_x = std::max(max_x, feature.xy.x());
        max_y = std::max(max_y, feature.xy.y());
      }

      if (frame_object_points.size() >= static_cast<usize>(min_points_per_view)) {
        object_points.push_back(frame_object_points);
        image_points.push_back(frame_image_points);
        feature_ids.push_back(frame_feature_ids);
        usable_imageset_indices.push_back(imageset_index);
        ++ stats.usable_views;
        stats.usable_points += frame_object_points.size();
      }
    }

    if (std::isfinite(min_x) && std::isfinite(max_x) && image_size.x() > 0 && image_size.y() > 0) {
      stats.bbox_area_ratio =
          std::max(0.f, max_x - min_x) * std::max(0.f, max_y - min_y) /
          static_cast<double>(image_size.x() * image_size.y());
    }

    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = std::max(image_size.x(), image_size.y());
    camera_matrix.at<double>(1, 1) = std::max(image_size.x(), image_size.y());
    camera_matrix.at<double>(0, 2) = 0.5 * image_size.x();
    camera_matrix.at<double>(1, 2) = 0.5 * image_size.y();
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    vector<cv::Mat> rvecs;
    vector<cv::Mat> tvecs;

    if (stats.usable_views < min_views) {
      stats.status = "insufficient";
      stats.reason = "usable_views_below_threshold";
    } else {
      try {
        const int flags = cv::CALIB_USE_INTRINSIC_GUESS;
        stats.rms = cv::calibrateCamera(
            object_points,
            image_points,
            cv::Size(image_size.x(), image_size.y()),
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            flags);
        stats.status = "solved";
        stats.reason = "";
      } catch (const cv::Exception& e) {
        stats.status = "failed";
        stats.reason = e.what();
      }
    }

    if (stats.status == "solved") {
      std::stringstream repo_name;
      repo_name << "intrinsics" << camera_index << "_" << entry.user_id << ".yaml";
      CentralOpenCVModel model(image_size.x(), image_size.y());
      FillCentralOpenCVModel(camera_matrix, dist_coeffs, &model);
      SaveCameraModel(model, (output_path / repo_name.str()).string().c_str());

      std::stringstream opencv_name;
      opencv_name << "opencv_intrinsics" << camera_index << "_" << entry.user_id << ".yaml";
      WriteOpenCVYaml(
          (output_path / opencv_name.str()).string(),
          image_size.x(),
          image_size.y(),
          camera_matrix,
          dist_coeffs,
          stats.rms);

      std::stringstream residual_name;
      residual_name << "residuals_camera" << camera_index << "_" << entry.user_id << ".tsv";
      std::ofstream residuals((output_path / residual_name.str()).string());
      residuals << "camera_index\tstage_name\tmachine\tuser_id\tview_index\timageset_index"
                << "\tfeature_id\tobserved_x\tobserved_y\tprojected_x\tprojected_y"
                << "\terror_x\terror_y\terror_norm\n";
      for (usize view_index = 0; view_index < object_points.size(); ++ view_index) {
        vector<cv::Point2f> projected_points;
        cv::projectPoints(
            object_points[view_index],
            rvecs[view_index],
            tvecs[view_index],
            camera_matrix,
            dist_coeffs,
            projected_points);
        for (usize point_index = 0; point_index < projected_points.size(); ++ point_index) {
          const cv::Point2f& observed = image_points[view_index][point_index];
          const cv::Point2f& projected = projected_points[point_index];
          const double error_x = projected.x - observed.x;
          const double error_y = projected.y - observed.y;
          const double error_norm = std::sqrt(error_x * error_x + error_y * error_y);
          residuals << camera_index << '\t' << entry.stage_name << '\t' << entry.machine << '\t'
                    << entry.user_id << '\t' << view_index << '\t'
                    << usable_imageset_indices[view_index] << '\t'
                    << feature_ids[view_index][point_index] << '\t'
                    << FormatDouble(observed.x) << '\t' << FormatDouble(observed.y) << '\t'
                    << FormatDouble(projected.x) << '\t' << FormatDouble(projected.y) << '\t'
                    << FormatDouble(error_x) << '\t' << FormatDouble(error_y) << '\t'
                    << FormatDouble(error_norm) << '\n';
        }
      }
    } else {
      insufficient << camera_index << '\t' << entry.stage_name << '\t' << entry.machine << '\t'
                   << entry.user_id << '\t' << stats.reason << '\t'
                   << stats.positive_views << '\t' << stats.usable_views << '\t'
                   << stats.total_points << '\t' << stats.usable_points << '\t'
                   << stats.max_points_per_view << '\t'
                   << FormatDouble(stats.bbox_area_ratio) << '\n';
    }

    summary << camera_index << '\t' << entry.stage_name << '\t' << entry.machine << '\t'
            << entry.user_id << '\t' << stats.status << '\t' << stats.reason << '\t'
            << image_size.x() << '\t' << image_size.y() << '\t'
            << stats.total_views << '\t' << stats.positive_views << '\t'
            << stats.usable_views << '\t' << stats.total_points << '\t'
            << stats.usable_points << '\t' << stats.max_points_per_view << '\t'
            << FormatDouble(stats.bbox_area_ratio) << '\t'
            << FormatDouble(stats.rms) << '\t';
    if (stats.status == "solved") {
      summary << FormatDouble(camera_matrix.at<double>(0, 0)) << '\t'
              << FormatDouble(camera_matrix.at<double>(1, 1)) << '\t'
              << FormatDouble(camera_matrix.at<double>(0, 2)) << '\t'
              << FormatDouble(camera_matrix.at<double>(1, 2)) << '\t'
              << FormatDouble(dist_coeffs.at<double>(0, 0)) << '\t'
              << FormatDouble(dist_coeffs.at<double>(1, 0)) << '\t'
              << FormatDouble(dist_coeffs.at<double>(2, 0)) << '\t'
              << FormatDouble(dist_coeffs.at<double>(3, 0)) << '\t'
              << FormatDouble(dist_coeffs.at<double>(4, 0)) << '\n';
    } else {
      summary << "\t\t\t\t\t\t\t\t\n";
    }
  }

  LOG(INFO) << "Wrote tower intrinsic summary to "
            << (output_path / "intrinsics_summary.tsv").string();
  LOG(INFO) << "Wrote insufficient camera list to "
            << (output_path / "insufficient_cameras.tsv").string();
  return EXIT_SUCCESS;
}

}  // namespace vis
