#include <open3d/Open3D.h>
#include <Eigen/Dense>

#include <iostream>
#include <chrono>
#include <filesystem>

// library headers
#include "pcd_block_estimation/utils.hpp"
#include "pcd_block_estimation/yaml_utils.hpp"
#include "pcd_block_estimation/template_utils.hpp"
#include "pcd_block_estimation/pose_estimation.hpp"

using namespace open3d;
using namespace pcd_block;
namespace fs = std::filesystem;

// ------------------------------------------------------------
// Parameters (still hardcoded – can become CLI / ROS params)
// ------------------------------------------------------------
static const std::string PCD_PLY = "../data/segmented_concrete_block.ply";
static const std::string TEMPLATE_DIR = "../data/templates";
static const std::string CALIB_YAML = "../data/calib_zed2i_to_seyond.yaml";

constexpr double DIST_THRESH = 0.02;
constexpr int MAX_PLANES = 3;
constexpr int MIN_INLIERS = 100;
constexpr double ICP_DIST = 0.04;

const Eigen::Vector3d Z_WORLD(0.0, -1.0, 0.0);
constexpr double ANGLE_THRESH =
  std::cos(30.0 * M_PI / 180.0);

// ------------------------------------------------------------
int main()
{


  // --------------------------------------------------------
  // Load & preprocess scene cloud
  // --------------------------------------------------------
  auto scene = std::make_shared<geometry::PointCloud>();
  if (!io::ReadPointCloud(PCD_PLY, *scene) || scene->IsEmpty()) {
    throw std::runtime_error("Failed to load scene point cloud");
  }

  // --------------------------------------------------------
  // Load calibration + transform cloud
  // --------------------------------------------------------
  Eigen::Matrix4d T_lidar_cam = load_T_4x4(CALIB_YAML);
  Eigen::Matrix4d T_cam_lidar = T_lidar_cam.inverse();

  Eigen::Matrix4d T_pitch = Eigen::Matrix4d::Identity();
  T_pitch.block<3, 3>(0, 0) =
    geometry::Geometry3D::GetRotationMatrixFromXYZ(
    Eigen::Vector3d(-M_PI / 6.0, 0, 0));


  auto templates = load_templates(TEMPLATE_DIR);

  auto t_start = std::chrono::high_resolution_clock::now();

  scene->RemoveStatisticalOutliers(20, 2.0);

  scene->Transform(T_pitch * T_cam_lidar);
  scene->RemoveStatisticalOutliers(20, 2.0);

  scene->EstimateNormals();

  // --------------------------------------------------------
  // Global Registration
  // --------------------------------------------------------

  GlobalRegistrationResult global_result = compute_global_registration(
    *scene,
    Z_WORLD,
    ANGLE_THRESH,
    MAX_PLANES,
    DIST_THRESH,
    MIN_INLIERS
  );

  // --------------------------------------------------------
  // Pose estimation (ICP yaw sweep)
  // --------------------------------------------------------
  LocalRegistrationResult result = compute_local_registration(
    *scene,
    templates,
    global_result,
    ICP_DIST,
    30       // yaw step in degrees
  );

  auto t_end = std::chrono::high_resolution_clock::now();

  // --------------------------------------------------------
  // Report result
  // --------------------------------------------------------
  std::cout << "\nBest template : " << result.template_name << "\n";
  std::cout << "Best yaw      : " << result.yaw_deg << " deg\n";
  std::cout << "Template idx  : " << result.template_index << "\n";
  std::cout << "Fitness       : " << result.icp.fitness_ << "\n";
  std::cout << "RMSE          : " << result.icp.inlier_rmse_ << "\n";

  std::cout << "Execution time: "
            << std::chrono::duration<double, std::milli>(
    t_end - t_start).count()
            << " ms\n";

  // --------------------------------------------------------
  // Visualization
  // --------------------------------------------------------
  auto best_vis =
    std::make_shared<geometry::PointCloud>(
    *templates[result.template_index].pcd);

  best_vis->Transform(result.icp.transformation_);
  best_vis->PaintUniformColor({0.0, 1.0, 0.0});

  auto frame =
    geometry::TriangleMesh::CreateCoordinateFrame(0.3);
  frame->Transform(result.icp.transformation_);

  visualization::DrawGeometries(
    {scene, best_vis, frame},
    "Final Pose Estimation Result",
    1200, 900
  );

  return 0;
}
