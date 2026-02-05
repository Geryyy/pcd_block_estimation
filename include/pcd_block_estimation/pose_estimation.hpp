#pragma once
#include "pcd_block_estimation/template_utils.hpp"
#include <Eigen/Dense>

namespace pcd_block
{

struct GlobalRegistrationResult
{
  bool success = false;
  Eigen::Vector3d center;     // scene centroid
  Eigen::Matrix3d R_base;     // roll/pitch fixed, yaw free
  int num_planes;
};

GlobalRegistrationResult compute_global_registration(
  const open3d::geometry::PointCloud & scene,
  const Eigen::Vector3d & z_world,
  double angle_thresh,
  int max_planes,
  double dist_thresh,
  int min_inliers
);


struct LocalRegistrationResult
{
  bool success = false;
  std::string template_name;
  int template_index = -1;
  int yaw_deg = 0;
  open3d::pipelines::registration::RegistrationResult icp;
};

LocalRegistrationResult compute_local_registration(
  const open3d::geometry::PointCloud & scene,
  const std::vector<TemplateData> & templates,
  const GlobalRegistrationResult global_registration,
  double icp_dist,
  int yaw_step_deg
);

Eigen::Matrix4d
globalResultToTransform(const GlobalRegistrationResult & glob)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

  if (!glob.success) {
    return T;
  }

  T.block<3, 3>(0, 0) = glob.R_base;
  T.block<3, 1>(0, 3) = glob.center;

  return T;
}

}
