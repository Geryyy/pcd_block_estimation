#pragma once

#include "pcd_block_estimation/template_utils.hpp"

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <limits>

namespace pcd_block
{

// ============================================================
// Global registration result (scene-only reasoning)
// ============================================================

struct GlobalRegistrationResult
{
  bool success = false;

  // Scene centroid
  Eigen::Vector3d center = Eigen::Vector3d::Zero();

  // Roll/pitch fixed, yaw-free base frame
  Eigen::Matrix3d R_base = Eigen::Matrix3d::Identity();

  // Raw plane info
  int num_planes = 0;
  std::vector<std::pair<Eigen::Vector4d,
                         open3d::geometry::PointCloud>> planes;

  // Concatenated inlier cloud of all planes
  std::shared_ptr<open3d::geometry::PointCloud> plane_cloud;

  // Identified top plane
  Eigen::Vector3d n_top = Eigen::Vector3d::Zero();
  Eigen::Vector3d c_top = Eigen::Vector3d::Zero();

  // Candidate front-facing planes (for 45° ambiguity)
  std::vector<Eigen::Vector3d> front_normals;
  std::vector<Eigen::Vector3d> front_centers;
};

// ------------------------------------------------------------

GlobalRegistrationResult compute_global_registration(
  const open3d::geometry::PointCloud & scene,
  const Eigen::Vector3d & z_world,
  double angle_thresh,
  int max_planes,
  double dist_thresh,
  int min_inliers,
  double max_plane_center_dist   // NEW
);

// ============================================================
// Local registration result (ICP refinement)
// ============================================================

constexpr double DEG2RAD = M_PI / 180.0;

struct HypothesisScore
{
  double geom = 0.0;   // plane compatibility
  double icp  = 0.0;   // ICP fitness
  double total = 0.0;
};


struct LocalRegistrationResult
{
  bool success = false;
  double score = -std::numeric_limits<double>::infinity();
  std::string template_name;
  int template_index = -1;

  // which hypothesis was chosen
  int front_plane_index = -1;   // index into front_normals
  int face_index = -1;          // 0 = long, 1 = short

  open3d::pipelines::registration::RegistrationResult icp;
};

// ------------------------------------------------------------

LocalRegistrationResult compute_local_registration(
  const open3d::geometry::PointCloud & scene,
  const std::vector<TemplateData> & templates,
  const GlobalRegistrationResult & global_registration,
  double icp_dist
);

// ------------------------------------------------------------
// Utility
// ------------------------------------------------------------

Eigen::Matrix4d
globalResultToTransform(const GlobalRegistrationResult & glob);

} // namespace pcd_block

