#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <chrono>

// #include "Timer.h"

// library headers
#include "pcd_block_estimation/yaml_utils.hpp"
#include "pcd_block_estimation/utils.hpp"
#include "pcd_block_estimation/mask_projection.hpp"
#include "pcd_block_estimation/template_utils.hpp"
#include "pcd_block_estimation/pose_estimation.hpp"

using namespace open3d;
using namespace pcd_block;

// ------------------------------------------------------------
// Parameters (hardcoded for now; easy to make CLI / ROS params)
// ------------------------------------------------------------
static const std::string PCL_PATH = "../data/pcl.ply";
static const std::string MASK_PATH = "../data/mask.png";
// static const std::string PCL_PATH = "../data/scene_1765975331_195477760.ply";
// static const std::string MASK_PATH = "../data/full_mask_1765975331_195477760.png";
static const std::string CALIB_YAML = "../data/calib_zed2i_to_seyond.yaml";
static const std::string TEMPLATE_DIR = "../data/templates";
static const std::string CUTOUT_PLY = "../data/segmented_concrete_block.ply";

constexpr double DIST_THRESH = 0.02;
constexpr int MAX_PLANES = 3;
constexpr int MIN_INLIERS = 100;
constexpr double ICP_DIST = 0.04;

const Eigen::Vector3d Z_WORLD(0.0, -1.0, 0.0);
constexpr double ANGLE_THRESH =
  std::cos(30.0 * M_PI / 180.0);
constexpr double MAX_PLANE_CENTER_DIST = 0.6; // meters

// ------------------------------------------------------------
int main()
{
  auto t_start = std::chrono::high_resolution_clock::now();


  // ========================================================
  // 1) MASK-BASED POINT CLOUD EXTRACTION
  // ========================================================

  // --------------------------------------------------------
  // Load mask + compute image center
  // --------------------------------------------------------
  cv::Mat mask = cv::imread(MASK_PATH, cv::IMREAD_GRAYSCALE);
  if (mask.empty()) {
    throw std::runtime_error("Failed to load mask");
  }

  Eigen::Vector2i uv_center = mask_center_uv(mask);
  std::cout << "Mask center (u,v): "
            << uv_center.transpose() << std::endl;

  // --------------------------------------------------------
  // Load calibration
  // --------------------------------------------------------
  YAML::Node calib = YAML::LoadFile(CALIB_YAML);

  Eigen::Matrix3d K;
  for (int i = 0; i < 9; ++i) {
    K(i / 3, i % 3) = calib["K"][i].as<double>();
  }

  Eigen::Matrix4d T_P_C = load_T_4x4(CALIB_YAML);

  // --------------------------------------------------------
  // Load raw point cloud
  // --------------------------------------------------------
  auto pcd_raw = std::make_shared<geometry::PointCloud>();
  if (!io::ReadPointCloud(PCL_PATH, *pcd_raw) || pcd_raw->IsEmpty()) {
    throw std::runtime_error("Failed to load raw point cloud");
  }

  std::cout << "Loaded raw cloud with "
            << pcd_raw->points_.size() << " points\n";

  auto t_start2 = std::chrono::high_resolution_clock::now();
  // --------------------------------------------------------
  // Mask-based cutout
  // --------------------------------------------------------
  auto pts_sel = select_points_by_mask(
    pcd_raw->points_, mask, K, T_P_C);

  if (pts_sel.empty()) {
    throw std::runtime_error("Mask-based cutout produced empty cloud");
  }

  std::cout << "Selected " << pts_sel.size()
            << " points inside mask\n";

  // --------------------------------------------------------
  // Save cutout for debugging / reuse
  // --------------------------------------------------------
  auto pcd_cutout = std::make_shared<geometry::PointCloud>();
  pcd_cutout->points_ = pts_sel;
  io::WritePointCloud(CUTOUT_PLY, *pcd_cutout);

  Eigen::Vector3d cutout_center = compute_center(pts_sel);
  std::cout << "Cutout centroid (x,y,z): "
            << cutout_center.transpose() << std::endl;

  auto t_end2 = std::chrono::high_resolution_clock::now();
  std::cout << "Coarse center computation time: "
            << std::chrono::duration<double, std::milli>(
    t_end2 - t_start2).count()
            << " ms\n";
  // ========================================================
  // 2) POSE ESTIMATION ON CUTOUT
  // ========================================================

  // --------------------------------------------------------
  // Camera pitch compensation + frame alignment
  // --------------------------------------------------------
  Eigen::Matrix4d T_lidar_cam = T_P_C;
  Eigen::Matrix4d T_cam_lidar = T_lidar_cam.inverse();

  Eigen::Matrix4d T_pitch = Eigen::Matrix4d::Identity();
  T_pitch.block<3, 3>(0, 0) =
    geometry::Geometry3D::GetRotationMatrixFromXYZ(
    Eigen::Vector3d(-M_PI / 6.0, 0, 0));

  pcd_cutout->RemoveStatisticalOutliers(20, 2.0);
  pcd_cutout->Transform(T_pitch * T_cam_lidar);
  pcd_cutout->RemoveStatisticalOutliers(20, 2.0);
  pcd_cutout->EstimateNormals();

  GlobalRegistrationResult globreg_result = compute_global_registration(
            *pcd_cutout,
            Z_WORLD,
            ANGLE_THRESH,
            MAX_PLANES,
            DIST_THRESH,
            MIN_INLIERS,
            MAX_PLANE_CENTER_DIST);

  // --------------------------------------------------------
  // Load templates
  // --------------------------------------------------------
  auto templates = load_templates(TEMPLATE_DIR);
  if (templates.empty()) {
    throw std::runtime_error("No templates loaded");
  }


  const auto & icp_scene =
    globreg_result.plane_cloud ?
    globreg_result.plane_cloud :
    pcd_cutout;
  // --------------------------------------------------------
  // Pose estimation (ICP yaw sweep)
  // --------------------------------------------------------
  LocalRegistrationResult result =
    compute_local_registration(
      *icp_scene,
      templates,
      globreg_result,
      ICP_DIST
    );

  auto t_end = std::chrono::high_resolution_clock::now();

  // ========================================================
  // REPORT + VISUALIZATION
  // ========================================================

  std::cout << "\nBest template : " << result.template_name << "\n";
  std::cout << "Template idx  : " << result.template_index << "\n";
  std::cout << "Fitness       : " << result.icp.fitness_ << "\n";
  std::cout << "RMSE          : " << result.icp.inlier_rmse_ << "\n";

  std::cout << "Total execution time: "
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
    {icp_scene, best_vis, frame},
    "Mask → Pose Estimation Pipeline",
    1200, 900
  );

  return 0;
}
