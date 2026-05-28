// Copyright 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the disclaimer in the documentation
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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "camera_calibration/bundle_adjustment/ba_state.h"
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

struct PnPView {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool solved = false;
  int point_count = 0;
  int inlier_count = 0;
  double mean_error = std::numeric_limits<double>::quiet_NaN();
  double median_error = std::numeric_limits<double>::quiet_NaN();
  SE3d camera_tr_board;
};

struct CameraPnPStats {
  int total_views = 0;
  int positive_views = 0;
  int solved_views = 0;
  int total_points = 0;
  int total_inliers = 0;
  vector<double> median_errors;
};

struct PairwiseEdge {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int camera_a = -1;
  int camera_b = -1;
  int shared_frames = 0;
  SE3d camera_a_tr_camera_b;
  double median_translation_residual = std::numeric_limits<double>::quiet_NaN();
  double median_rotation_residual_deg = std::numeric_limits<double>::quiet_NaN();
};

using PnPViewVector = vector<PnPView, Eigen::aligned_allocator<PnPView>>;
using PairwiseEdgeVector = vector<PairwiseEdge, Eigen::aligned_allocator<PairwiseEdge>>;

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

double Median(vector<double> values) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const usize mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  double result = values[mid];
  if ((values.size() % 2) == 0) {
    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    result = 0.5 * (result + values[mid - 1]);
  }
  return result;
}

string FindIntrinsicPath(const string& intrinsics_directory, int camera_index) {
  const boost::filesystem::path dir(intrinsics_directory);
  std::stringstream exact_name;
  exact_name << "intrinsics" << camera_index << ".yaml";
  boost::filesystem::path exact_path = dir / exact_name.str();
  if (boost::filesystem::exists(exact_path)) {
    return exact_path.string();
  }

  std::stringstream prefix_stream;
  prefix_stream << "intrinsics" << camera_index << "_";
  const string prefix = prefix_stream.str();
  if (!boost::filesystem::exists(dir)) {
    return "";
  }

  vector<string> candidates;
  for (boost::filesystem::directory_iterator it(dir), end; it != end; ++ it) {
    if (!boost::filesystem::is_regular_file(*it)) {
      continue;
    }
    const string filename = it->path().filename().string();
    if (filename.compare(0, prefix.size(), prefix) == 0 &&
        it->path().extension() == ".yaml") {
      candidates.push_back(it->path().string());
    }
  }
  std::sort(candidates.begin(), candidates.end());
  return candidates.empty() ? "" : candidates.front();
}

bool BuildOpenCVCamera(
    const CentralOpenCVModel& model,
    cv::Mat* camera_matrix,
    cv::Mat* dist_coeffs) {
  const auto& p = model.parameters();
  if (p.size() < 12) {
    return false;
  }

  *camera_matrix = cv::Mat::eye(3, 3, CV_64F);
  camera_matrix->at<double>(0, 0) = p(0);
  camera_matrix->at<double>(1, 1) = p(1);
  camera_matrix->at<double>(0, 2) = p(2);
  camera_matrix->at<double>(1, 2) = p(3);

  *dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
  dist_coeffs->at<double>(0, 0) = p(4);
  dist_coeffs->at<double>(1, 0) = p(5);
  dist_coeffs->at<double>(2, 0) = p(10);
  dist_coeffs->at<double>(3, 0) = p(11);
  dist_coeffs->at<double>(4, 0) = p(6);
  dist_coeffs->at<double>(5, 0) = p(7);
  dist_coeffs->at<double>(6, 0) = p(8);
  dist_coeffs->at<double>(7, 0) = p(9);
  return true;
}

SE3d PoseFromCv(const cv::Mat& rvec, const cv::Mat& tvec) {
  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);

  Mat3d rotation;
  for (int y = 0; y < 3; ++ y) {
    for (int x = 0; x < 3; ++ x) {
      rotation(y, x) = rotation_cv.at<double>(y, x);
    }
  }

  Vec3d translation(
      tvec.at<double>(0, 0),
      tvec.at<double>(1, 0),
      tvec.at<double>(2, 0));
  return SE3d(rotation, translation);
}

bool FeatureToPoint(
    const KnownGeometry& geometry,
    int feature_id,
    int corner_id_offset,
    Vec3f* point) {
  if (corner_id_offset == 0 && geometry.GetFeaturePoint3D(feature_id, point)) {
    return true;
  }

  const int tag_id = feature_id / 4;
  const int corner_id = feature_id % 4;
  const int geometry_feature_id =
      tag_id * 4 + ((corner_id + corner_id_offset) % 4);
  if (geometry.GetFeaturePoint3D(geometry_feature_id, point)) {
    return true;
  }

  if (corner_id_offset != 0 && geometry.GetFeaturePoint3D(feature_id, point)) {
    return true;
  }
  return false;
}

void BuildViewCorrespondences(
    const vector<PointFeature>& features,
    const KnownGeometry& geometry,
    int corner_id_offset,
    vector<cv::Point3f>* object_points,
    vector<cv::Point2f>* image_points) {
  object_points->clear();
  image_points->clear();

  for (const PointFeature& feature : features) {
    Vec3f point;
    if (!FeatureToPoint(geometry, feature.id, corner_id_offset, &point)) {
      continue;
    }
    object_points->emplace_back(point.x(), point.y(), point.z());
    image_points->emplace_back(feature.xy.x(), feature.xy.y());
  }
}

bool SolveViewPnP(
    const vector<cv::Point3f>& object_points,
    const vector<cv::Point2f>& image_points,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    int min_points_per_view,
    double ransac_reprojection_threshold,
    PnPView* view) {
  view->point_count = object_points.size();
  if (object_points.size() < static_cast<usize>(min_points_per_view)) {
    return false;
  }

  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
  vector<int> inliers;
  bool ok = false;
  try {
    ok = cv::solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        false,
        100,
        ransac_reprojection_threshold,
        0.99,
        inliers,
        cv::SOLVEPNP_ITERATIVE);
  } catch (const cv::Exception& e) {
    LOG(WARNING) << "solvePnPRansac failed: " << e.what();
    return false;
  }

  if (!ok || inliers.size() < static_cast<usize>(min_points_per_view)) {
    return false;
  }

  vector<cv::Point3f> inlier_object_points;
  vector<cv::Point2f> inlier_image_points;
  inlier_object_points.reserve(inliers.size());
  inlier_image_points.reserve(inliers.size());
  for (int index : inliers) {
    inlier_object_points.push_back(object_points[index]);
    inlier_image_points.push_back(image_points[index]);
  }

  try {
    cv::solvePnP(
        inlier_object_points,
        inlier_image_points,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        true,
        cv::SOLVEPNP_ITERATIVE);
  } catch (const cv::Exception& e) {
    LOG(WARNING) << "solvePnP refinement failed: " << e.what();
    return false;
  }

  vector<cv::Point2f> projected_points;
  cv::projectPoints(
      inlier_object_points,
      rvec,
      tvec,
      camera_matrix,
      dist_coeffs,
      projected_points);

  vector<double> errors;
  errors.reserve(projected_points.size());
  double error_sum = 0;
  for (usize i = 0; i < projected_points.size(); ++ i) {
    const double dx = projected_points[i].x - inlier_image_points[i].x;
    const double dy = projected_points[i].y - inlier_image_points[i].y;
    const double error = std::sqrt(dx * dx + dy * dy);
    errors.push_back(error);
    error_sum += error;
  }

  view->solved = true;
  view->inlier_count = inliers.size();
  view->mean_error = error_sum / std::max<usize>(1, errors.size());
  view->median_error = Median(errors);
  view->camera_tr_board = PoseFromCv(rvec, tvec);
  return true;
}

SE3d AveragePose(const vector<SE3d>& poses) {
  CHECK(!poses.empty());

  Mat4d q_outer = Mat4d::Zero();
  vector<double> tx;
  vector<double> ty;
  vector<double> tz;
  tx.reserve(poses.size());
  ty.reserve(poses.size());
  tz.reserve(poses.size());

  for (const SE3d& pose : poses) {
    const Quaterniond q = pose.unit_quaternion();
    Vec4d q_vec(q.w(), q.x(), q.y(), q.z());
    q_outer += q_vec * q_vec.transpose();
    tx.push_back(pose.translation().x());
    ty.push_back(pose.translation().y());
    tz.push_back(pose.translation().z());
  }

  Eigen::SelfAdjointEigenSolver<Mat4d> solver(q_outer);
  Vec4d q_vec = solver.eigenvectors().col(3);
  Quaterniond q(q_vec(0), q_vec(1), q_vec(2), q_vec(3));
  q.normalize();
  if (q.w() < 0) {
    q.coeffs() *= -1;
  }

  return SE3d(q.toRotationMatrix(), Vec3d(Median(tx), Median(ty), Median(tz)));
}

double RotationAngleDeg(const SE3d& pose) {
  return 180.0 / M_PI * Eigen::AngleAxisd(pose.unit_quaternion()).angle();
}

PairwiseEdge BuildPairwiseEdge(
    int camera_a,
    int camera_b,
    const vector<PnPViewVector>& pnp_views,
    int min_shared_views) {
  vector<SE3d> votes;
  const int image_count = pnp_views[camera_a].size();
  for (int imageset_index = 0; imageset_index < image_count; ++ imageset_index) {
    const PnPView& view_a = pnp_views[camera_a][imageset_index];
    const PnPView& view_b = pnp_views[camera_b][imageset_index];
    if (!view_a.solved || !view_b.solved) {
      continue;
    }
    votes.push_back(view_a.camera_tr_board * view_b.camera_tr_board.inverse());
  }

  PairwiseEdge edge;
  edge.camera_a = camera_a;
  edge.camera_b = camera_b;
  edge.shared_frames = votes.size();
  if (votes.size() < static_cast<usize>(min_shared_views)) {
    return edge;
  }

  edge.camera_a_tr_camera_b = AveragePose(votes);
  vector<double> translation_residuals;
  vector<double> rotation_residuals;
  translation_residuals.reserve(votes.size());
  rotation_residuals.reserve(votes.size());
  for (const SE3d& vote : votes) {
    SE3d delta = edge.camera_a_tr_camera_b.inverse() * vote;
    translation_residuals.push_back(delta.translation().norm());
    rotation_residuals.push_back(RotationAngleDeg(delta));
  }
  edge.median_translation_residual = Median(translation_residuals);
  edge.median_rotation_residual_deg = Median(rotation_residuals);
  return edge;
}

bool EstimateCameraRigFromEdges(
    int camera_count,
    int reference_camera,
    const PairwiseEdgeVector& edges,
    vector<bool>* camera_used,
    vector<SE3d>* camera_tr_rig) {
  camera_used->assign(camera_count, false);
  camera_tr_rig->assign(camera_count, SE3d());
  camera_used->at(reference_camera) = true;
  camera_tr_rig->at(reference_camera) = SE3d();

  bool progress = true;
  while (progress) {
    progress = false;
    for (const PairwiseEdge& edge : edges) {
      if (edge.shared_frames <= 0) {
        continue;
      }

      if (camera_used->at(edge.camera_b) && !camera_used->at(edge.camera_a)) {
        camera_tr_rig->at(edge.camera_a) =
            edge.camera_a_tr_camera_b * camera_tr_rig->at(edge.camera_b);
        camera_used->at(edge.camera_a) = true;
        progress = true;
      } else if (camera_used->at(edge.camera_a) && !camera_used->at(edge.camera_b)) {
        camera_tr_rig->at(edge.camera_b) =
            edge.camera_a_tr_camera_b.inverse() * camera_tr_rig->at(edge.camera_a);
        camera_used->at(edge.camera_b) = true;
        progress = true;
      }
    }
  }

  return std::all_of(camera_used->begin(), camera_used->end(), [](bool used) {
    return used;
  });
}

void EstimateBoardPosesInRig(
    const vector<PnPViewVector>& pnp_views,
    const vector<bool>& camera_used,
    const vector<SE3d>& camera_tr_rig,
    vector<bool>* image_used,
    vector<SE3d>* rig_tr_board) {
  const int camera_count = pnp_views.size();
  const int image_count = pnp_views.empty() ? 0 : pnp_views[0].size();
  image_used->assign(image_count, false);
  rig_tr_board->assign(image_count, SE3d());

  for (int imageset_index = 0; imageset_index < image_count; ++ imageset_index) {
    vector<SE3d> votes;
    for (int camera_index = 0; camera_index < camera_count; ++ camera_index) {
      if (!camera_used[camera_index] || !pnp_views[camera_index][imageset_index].solved) {
        continue;
      }
      votes.push_back(
          camera_tr_rig[camera_index].inverse() *
          pnp_views[camera_index][imageset_index].camera_tr_board);
    }
    if (!votes.empty()) {
      (*image_used)[imageset_index] = true;
      (*rig_tr_board)[imageset_index] = AveragePose(votes);
    }
  }
}

void PopulateKnownPoints(
    const Dataset& dataset,
    const KnownGeometry& geometry,
    int corner_id_offset,
    BAState* state) {
  state->points.clear();
  state->feature_id_to_points_index.clear();

  for (int imageset_index = 0; imageset_index < dataset.ImagesetCount(); ++ imageset_index) {
    shared_ptr<const Imageset> imageset = dataset.GetImageset(imageset_index);
    for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
      for (const PointFeature& feature : imageset->FeaturesOfCamera(camera_index)) {
        if (state->feature_id_to_points_index.find(feature.id) !=
            state->feature_id_to_points_index.end()) {
          continue;
        }
        Vec3f point;
        if (!FeatureToPoint(geometry, feature.id, corner_id_offset, &point)) {
          continue;
        }
        const int point_index = state->points.size();
        state->feature_id_to_points_index[feature.id] = point_index;
        state->points.emplace_back(point.x(), point.y(), point.z());
      }
    }
  }
}

}  // namespace

int EstimateFixedIntrinsicRig(
    const string& dataset_path,
    const string& intrinsics_directory,
    const string& camera_manifest_path,
    const string& output_directory,
    int min_points_per_view,
    int min_shared_views,
    int reference_camera,
    double pnp_reprojection_threshold,
    int corner_id_offset) {
  if (dataset_path.empty() || intrinsics_directory.empty() || output_directory.empty()) {
    LOG(ERROR) << "--estimate_fixed_intrinsic_rig requires --dataset_files, "
               << "--fixed_intrinsics_directory, and --output_directory.";
    return EXIT_FAILURE;
  }
  if (min_points_per_view < 4) {
    LOG(ERROR) << "--fixed_rig_min_points_per_view must be at least 4.";
    return EXIT_FAILURE;
  }
  if (min_shared_views < 1) {
    LOG(ERROR) << "--fixed_rig_min_shared_views must be positive.";
    return EXIT_FAILURE;
  }
  if (pnp_reprojection_threshold <= 0) {
    LOG(ERROR) << "--fixed_rig_pnp_reprojection_threshold must be positive.";
    return EXIT_FAILURE;
  }
  if (corner_id_offset < 0 || corner_id_offset > 3) {
    LOG(ERROR) << "--fixed_rig_corner_id_offset must be in [0, 3].";
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
  if (reference_camera < 0 || reference_camera >= dataset.num_cameras()) {
    LOG(ERROR) << "--fixed_rig_reference_camera is outside the dataset camera range.";
    return EXIT_FAILURE;
  }

  boost::filesystem::create_directories(output_directory);
  const boost::filesystem::path output_path(output_directory);
  const unordered_map<int, CameraManifestEntry> manifest =
      LoadCameraManifest(camera_manifest_path);
  const KnownGeometry& geometry = dataset.GetKnownGeometry(0);

  vector<shared_ptr<CameraModel>> intrinsics(dataset.num_cameras());
  vector<cv::Mat> camera_matrices(dataset.num_cameras());
  vector<cv::Mat> dist_coeffs(dataset.num_cameras());
  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    const string intrinsic_path = FindIntrinsicPath(intrinsics_directory, camera_index);
    if (intrinsic_path.empty()) {
      LOG(ERROR) << "Cannot find intrinsics for camera " << camera_index
                 << " in " << intrinsics_directory;
      return EXIT_FAILURE;
    }

    intrinsics[camera_index] = LoadCameraModel(intrinsic_path.c_str());
    CentralOpenCVModel* opencv_model =
        dynamic_cast<CentralOpenCVModel*>(intrinsics[camera_index].get());
    if (!opencv_model) {
      LOG(ERROR) << "Fixed-rig initializer currently requires CentralOpenCVModel intrinsics: "
                 << intrinsic_path;
      return EXIT_FAILURE;
    }
    if (!BuildOpenCVCamera(*opencv_model, &camera_matrices[camera_index], &dist_coeffs[camera_index])) {
      LOG(ERROR) << "Could not convert intrinsics for camera " << camera_index;
      return EXIT_FAILURE;
    }
  }

  vector<PnPViewVector> pnp_views(
      dataset.num_cameras(),
      PnPViewVector(dataset.ImagesetCount()));
  vector<CameraPnPStats> camera_stats(dataset.num_cameras());

  std::ofstream pnp_file((output_path / "pnp_views.tsv").string());
  pnp_file << "camera_index\tstage_name\tmachine\tuser_id\timageset_index\tfilename"
           << "\tstatus\tpoints\tinliers\tmean_error_px\tmedian_error_px"
           << "\ttx\tty\ttz\tqx\tqy\tqz\tqw\n";

  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    CameraManifestEntry entry = DefaultManifestEntry(camera_index);
    auto manifest_it = manifest.find(camera_index);
    if (manifest_it != manifest.end()) {
      entry = manifest_it->second;
    }

    CameraPnPStats& stats = camera_stats[camera_index];
    stats.total_views = dataset.ImagesetCount();

    for (int imageset_index = 0; imageset_index < dataset.ImagesetCount(); ++ imageset_index) {
      shared_ptr<const Imageset> imageset = dataset.GetImageset(imageset_index);
      const vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);
      if (!features.empty()) {
        ++ stats.positive_views;
      }
      stats.total_points += features.size();

      vector<cv::Point3f> object_points;
      vector<cv::Point2f> image_points;
      BuildViewCorrespondences(
          features,
          geometry,
          corner_id_offset,
          &object_points,
          &image_points);

      PnPView view;
      const bool solved = SolveViewPnP(
          object_points,
          image_points,
          camera_matrices[camera_index],
          dist_coeffs[camera_index],
          min_points_per_view,
          pnp_reprojection_threshold,
          &view);
      pnp_views[camera_index][imageset_index] = view;

      if (solved) {
        ++ stats.solved_views;
        stats.total_inliers += view.inlier_count;
        stats.median_errors.push_back(view.median_error);
      }

      const SE3d& pose = view.camera_tr_board;
      pnp_file << camera_index << '\t' << entry.stage_name << '\t' << entry.machine << '\t'
               << entry.user_id << '\t' << imageset_index << '\t'
               << imageset->GetFilename() << '\t'
               << (solved ? "solved" : "failed") << '\t'
               << view.point_count << '\t' << view.inlier_count << '\t'
               << FormatDouble(view.mean_error) << '\t'
               << FormatDouble(view.median_error) << '\t'
               << FormatDouble(pose.translation().x()) << '\t'
               << FormatDouble(pose.translation().y()) << '\t'
               << FormatDouble(pose.translation().z()) << '\t'
               << FormatDouble(pose.unit_quaternion().x()) << '\t'
               << FormatDouble(pose.unit_quaternion().y()) << '\t'
               << FormatDouble(pose.unit_quaternion().z()) << '\t'
               << FormatDouble(pose.unit_quaternion().w()) << '\n';
    }
  }

  PairwiseEdgeVector edges;
  for (int camera_a = 0; camera_a < dataset.num_cameras(); ++ camera_a) {
    for (int camera_b = 0; camera_b < dataset.num_cameras(); ++ camera_b) {
      if (camera_a == camera_b) {
        continue;
      }
      PairwiseEdge edge = BuildPairwiseEdge(
          camera_a,
          camera_b,
          pnp_views,
          min_shared_views);
      if (edge.shared_frames >= min_shared_views) {
        edges.push_back(edge);
      }
    }
  }
  std::sort(edges.begin(), edges.end(), [](const PairwiseEdge& a, const PairwiseEdge& b) {
    if (a.shared_frames != b.shared_frames) {
      return a.shared_frames > b.shared_frames;
    }
    return a.median_translation_residual < b.median_translation_residual;
  });

  vector<bool> camera_used;
  vector<SE3d> camera_tr_rig;
  const bool all_cameras_connected = EstimateCameraRigFromEdges(
      dataset.num_cameras(),
      reference_camera,
      edges,
      &camera_used,
      &camera_tr_rig);

  vector<bool> image_used;
  vector<SE3d> rig_tr_board;
  EstimateBoardPosesInRig(
      pnp_views,
      camera_used,
      camera_tr_rig,
      &image_used,
      &rig_tr_board);

  BAState state;
  state.image_used = image_used;
  state.camera_tr_rig = camera_tr_rig;
  state.rig_tr_global = rig_tr_board;
  state.intrinsics = intrinsics;
  PopulateKnownPoints(dataset, geometry, corner_id_offset, &state);

  if (!SaveBAState(output_directory.c_str(), state)) {
    LOG(ERROR) << "Could not save fixed-intrinsic rig state to " << output_directory;
    return EXIT_FAILURE;
  }
  SavePoses(camera_used, camera_tr_rig, (output_path / "camera_tr_rig_used.yaml").string().c_str());

  std::ofstream camera_summary((output_path / "camera_pnp_summary.tsv").string());
  camera_summary << "camera_index\tstage_name\tmachine\tuser_id\tconnected"
                 << "\ttotal_views\tpositive_views\tsolved_views\ttotal_points"
                 << "\ttotal_inliers\tmedian_view_error_px\ttx\tty\ttz\tqx\tqy\tqz\tqw\n";
  for (int camera_index = 0; camera_index < dataset.num_cameras(); ++ camera_index) {
    CameraManifestEntry entry = DefaultManifestEntry(camera_index);
    auto manifest_it = manifest.find(camera_index);
    if (manifest_it != manifest.end()) {
      entry = manifest_it->second;
    }

    const CameraPnPStats& stats = camera_stats[camera_index];
    const SE3d& pose = camera_tr_rig[camera_index];
    camera_summary << camera_index << '\t' << entry.stage_name << '\t' << entry.machine << '\t'
                   << entry.user_id << '\t' << (camera_used[camera_index] ? "yes" : "no") << '\t'
                   << stats.total_views << '\t' << stats.positive_views << '\t'
                   << stats.solved_views << '\t' << stats.total_points << '\t'
                   << stats.total_inliers << '\t'
                   << FormatDouble(Median(stats.median_errors)) << '\t'
                   << FormatDouble(pose.translation().x()) << '\t'
                   << FormatDouble(pose.translation().y()) << '\t'
                   << FormatDouble(pose.translation().z()) << '\t'
                   << FormatDouble(pose.unit_quaternion().x()) << '\t'
                   << FormatDouble(pose.unit_quaternion().y()) << '\t'
                   << FormatDouble(pose.unit_quaternion().z()) << '\t'
                   << FormatDouble(pose.unit_quaternion().w()) << '\n';
  }

  std::ofstream edge_file((output_path / "pairwise_edges.tsv").string());
  edge_file << "camera_a\tcamera_b\tshared_frames\tmedian_translation_residual_m"
            << "\tmedian_rotation_residual_deg\ttx\tty\ttz\tqx\tqy\tqz\tqw\n";
  for (const PairwiseEdge& edge : edges) {
    const SE3d& pose = edge.camera_a_tr_camera_b;
    edge_file << edge.camera_a << '\t' << edge.camera_b << '\t'
              << edge.shared_frames << '\t'
              << FormatDouble(edge.median_translation_residual) << '\t'
              << FormatDouble(edge.median_rotation_residual_deg) << '\t'
              << FormatDouble(pose.translation().x()) << '\t'
              << FormatDouble(pose.translation().y()) << '\t'
              << FormatDouble(pose.translation().z()) << '\t'
              << FormatDouble(pose.unit_quaternion().x()) << '\t'
              << FormatDouble(pose.unit_quaternion().y()) << '\t'
              << FormatDouble(pose.unit_quaternion().z()) << '\t'
              << FormatDouble(pose.unit_quaternion().w()) << '\n';
  }

  const int connected_count = std::count(camera_used.begin(), camera_used.end(), true);
  const int localized_imagesets = std::count(image_used.begin(), image_used.end(), true);
  LOG(INFO) << "Fixed-intrinsic rig initialization connected "
            << connected_count << " / " << dataset.num_cameras() << " cameras.";
  LOG(INFO) << "Localized " << localized_imagesets << " / "
            << dataset.ImagesetCount() << " synchronized board poses.";
  LOG(INFO) << "Wrote fixed-intrinsic rig state to " << output_directory;
  return all_cameras_connected ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace vis
