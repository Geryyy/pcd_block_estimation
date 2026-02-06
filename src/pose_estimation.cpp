#include "pcd_block_estimation/pose_estimation.hpp"
#include "pcd_block_estimation/utils.hpp"

using namespace open3d;

namespace pcd_block
{


GlobalRegistrationResult compute_global_registration(
  const geometry::PointCloud & scene,
  const Eigen::Vector3d & Z_WORLD,
  double angle_thresh,
  int max_planes,
  double dist_thresh,
  int min_inliers
)
{
  GlobalRegistrationResult result;
  result.center = compute_center(scene);

  // --------------------------------------------------------
  // Plane extraction
  // --------------------------------------------------------
  auto planes = extract_planes(
    scene, max_planes, dist_thresh, min_inliers);

  if (planes.empty()) {
    std::cerr << "No planes detected" << std::endl;
    result.success = false;
    return result;
  }

  result.num_planes = planes.size();
  result.planes = planes;

  // --------------------------------------------------------
  // Find top plane normal and store plane points
  // --------------------------------------------------------
  auto plane_cloud = std::make_shared<geometry::PointCloud>();

  Eigen::Vector3d n_top;
  bool found = false;

  for (const auto & [plane, pc] : planes) {
    *plane_cloud += pc;   // concatenate inliers

    Eigen::Vector3d n = plane.head<3>().normalized();
    if (n.dot(Z_WORLD) < 0.0) {
      n = -n;
    }

    if (std::abs(n.dot(Z_WORLD)) > angle_thresh) {
      n_top = n;
      found = true;
      break;
    }
  }

  result.plane_cloud = plane_cloud;

  if (!found) {
    std::cerr << "Top plane not detected" << std::endl;
    result.success = false;
    return result;
  }

  // --------------------------------------------------------
  // Build base frame (yaw-free)
  // --------------------------------------------------------
  Eigen::Vector3d z_cam = n_top.normalized();

  Eigen::Vector3d tmp(1.0, 0.0, 0.0);
  if (std::abs(tmp.dot(z_cam)) > 0.9) {
    tmp = Eigen::Vector3d(0.0, 1.0, 0.0);
  }

  Eigen::Vector3d x_base =
    (tmp - tmp.dot(z_cam) * z_cam).normalized();
  Eigen::Vector3d y_base =
    z_cam.cross(x_base).normalized();

  result.R_base.col(0) = x_base;
  result.R_base.col(1) = y_base;
  result.R_base.col(2) = z_cam;

  result.success = true;

  return result;
}


LocalRegistrationResult compute_local_registration(
  const geometry::PointCloud & scene,
  const std::vector<TemplateData> & templates,
  const GlobalRegistrationResult global_registration,
  double icp_dist,
  int yaw_step_deg)
{
  LocalRegistrationResult best{};
  best.icp.fitness_ = -1.0;

  auto scene_center = global_registration.center;
  auto R_base = global_registration.R_base;
  int num_planes = global_registration.num_planes;

  int idx = 0;
  for (const auto & tpl : templates) {
    if (tpl.num_faces != num_planes) {
      idx++; continue;
    }

    Eigen::Vector3d tpl_center =
      compute_center(*tpl.pcd);

    for (int yaw = 0; yaw < 360; yaw += yaw_step_deg) {

      Eigen::Matrix3d R_yaw =
        geometry::Geometry3D::GetRotationMatrixFromAxisAngle(
        R_base.col(2) * yaw * M_PI / 180.0);

      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      T.block<3, 3>(0, 0) = R_yaw * R_base;
      T.block<3, 1>(0, 3) =
        scene_center - T.block<3, 3>(0, 0) * tpl_center;

      auto result =
        pipelines::registration::RegistrationICP(
        *tpl.pcd, scene,
        icp_dist, T,
        pipelines::registration::
        TransformationEstimationPointToPlane());

      if (result.fitness_ > best.icp.fitness_) {
        best = {true, tpl.name, idx, yaw, result};
      }
    }
    idx++;
  }
  return best;
}


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
