#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <unordered_map>
#include <iostream>
#include <chrono>
#include <sstream>

// library headers
#include "pcd_block_estimation/yaml_utils.hpp"
#include "pcd_block_estimation/utils.hpp"
#include "pcd_block_estimation/mask_projection.hpp"
#include "pcd_block_estimation/template_utils.hpp"
#include "pcd_block_estimation/pose_estimation.hpp"

using namespace open3d;
using namespace pcd_block;
namespace fs = std::filesystem;

// ============================================================
// MODE SWITCH
// ============================================================
enum class PipelineMode {
  GLOBAL_ONLY,
  FULL_PIPELINE
};

// static constexpr PipelineMode MODE = PipelineMode::FULL_PIPELINE;
static constexpr PipelineMode MODE = PipelineMode::GLOBAL_ONLY;

// ============================================================
// Paths
// ============================================================
static const std::string DATA_DIR     = "../data/dump";
static const std::string CALIB_YAML   = "../data/calib_zed2i_to_seyond.yaml";
static const std::string TEMPLATE_DIR = "../data/templates";
static const std::string OUT_DIR      = "../data/debug";

// ============================================================
// Parameters
// ============================================================
constexpr double DIST_THRESH = 0.02;
constexpr int    MAX_PLANES  = 3;
constexpr int    MIN_INLIERS = 100;
constexpr double ICP_DIST   = 0.04;

const Eigen::Vector3d Z_WORLD(0.0, -1.0, 0.0);
constexpr double ANGLE_THRESH =
  std::cos(30.0 * M_PI / 180.0);

// ============================================================
// Helpers
// ============================================================
static Eigen::Vector3d plane_color(int i)
{
  static const std::vector<Eigen::Vector3d> colors = {
    {1.0, 0.0, 0.0}, // red
    {0.0, 1.0, 0.0}, // green
    {0.0, 0.0, 1.0}, // blue
    {1.0, 1.0, 0.0}, // yellow
    {1.0, 0.0, 1.0}, // magenta
    {0.0, 1.0, 1.0}  // cyan
  };
  return colors[i % colors.size()];
}

static std::shared_ptr<geometry::TriangleMesh>
make_normal_cylinder(
  const Eigen::Vector3d & origin,
  const Eigen::Vector3d & normal,
  const Eigen::Vector3d & color,
  double length = 0.3,
  double radius = 0.01)
{
  auto cyl =
    geometry::TriangleMesh::CreateCylinder(radius, length);
  cyl->PaintUniformColor(color);

  Eigen::Vector3d n = normal.normalized();

  // Rotate cylinder Z-axis onto normal
  Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
  Eigen::Matrix3d R =
    Eigen::Quaterniond::FromTwoVectors(z_axis, n)
      .toRotationMatrix();

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3,3>(0,0) = R;

  // place cylinder so it starts at origin and points outward
  T.block<3,1>(0,3) = origin + 0.5 * length * n;

  cyl->Transform(T);
  cyl->ComputeVertexNormals();

  return cyl;
}



static std::string extract_timestamp(const fs::path & p)
{
  // 1765974788_298710784_cloud.ply
  // 1765974788_298710784_mask.png
  const std::string stem = p.stem().string();

  std::vector<std::string> tokens;
  std::stringstream ss(stem);
  std::string item;

  while (std::getline(ss, item, '_'))
    tokens.push_back(item);

  if (tokens.size() < 3)
    return {};

  return tokens[0] + "_" + tokens[1];
}

// ------------------------------------------------------------
static void visualize_global(
  const std::shared_ptr<geometry::PointCloud> & cutout,
  const GlobalRegistrationResult & globreg,
  const std::string & title)
{
  std::vector<std::shared_ptr<const geometry::Geometry>> geoms;

  // --------------------------------------------------------
  // Cutout (background)
  // --------------------------------------------------------
  auto cut = std::make_shared<geometry::PointCloud>(*cutout);
  cut->PaintUniformColor({0.6, 0.6, 0.6});
  geoms.push_back(cut);

  // --------------------------------------------------------
  // Planes + normals
  // --------------------------------------------------------
  for (size_t i = 0; i < globreg.planes.size(); ++i) {

    const auto & plane = globreg.planes[i];
    const Eigen::Vector4d & eq = plane.first;
    const auto & plane_cloud = plane.second;

    Eigen::Vector3d color = plane_color(i);

    // Plane cloud
    auto plane_vis =
      std::make_shared<geometry::PointCloud>(plane_cloud);
    plane_vis->PaintUniformColor(color);
    geoms.push_back(plane_vis);

    // Plane centroid
    Eigen::Vector3d centroid =
      compute_center(plane_cloud.points_);

    // Plane normal from ax + by + cz + d = 0
    Eigen::Vector3d normal(eq[0], eq[1], eq[2]);

    // Optional: enforce consistent orientation
    if (normal.dot(Z_WORLD) < 0.0)
      normal = -normal;

    // Normal as cylinder
    geoms.push_back(
      make_normal_cylinder(
        centroid,
        normal,
        color,
        0.3,   // length
        0.01   // radius
      ));
  }

  // --------------------------------------------------------
  // Global frame
  // --------------------------------------------------------
  if (globreg.success) {
    auto frame =
      geometry::TriangleMesh::CreateCoordinateFrame(0.3);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,1>(0,3) = globreg.center;
    frame->Transform(T);
    geoms.push_back(frame);
  }

  visualization::DrawGeometries(
    geoms,
    title,
    1200,
    900);
}

// ------------------------------------------------------------
static void visualize_full(
  const std::shared_ptr<geometry::PointCloud> & scene,
  const TemplateData & tmpl,
  const LocalRegistrationResult & result,
  const std::string & title)
{
  // Scene in GREEN
  auto scene_vis =
    std::make_shared<geometry::PointCloud>(*scene);
  scene_vis->PaintUniformColor({0.0, 1.0, 0.0});

  // Fitted template in RED
  auto tmpl_vis =
    std::make_shared<geometry::PointCloud>(*tmpl.pcd);
  tmpl_vis->Transform(result.icp.transformation_);
  tmpl_vis->PaintUniformColor({1.0, 0.0, 0.0});

  auto frame =
    geometry::TriangleMesh::CreateCoordinateFrame(0.3);
  frame->Transform(result.icp.transformation_);

  std::vector<std::shared_ptr<const geometry::Geometry>> geoms;
  geoms.push_back(scene_vis);
  geoms.push_back(tmpl_vis);
  geoms.push_back(frame);

  visualization::DrawGeometries(
    geoms,
    title,
    1200,
    900);
}


// ============================================================
// MAIN
// ============================================================
int main()
{
  fs::create_directories(OUT_DIR);

  // ----------------------------------------------------------
  // Load calibration
  // ----------------------------------------------------------
  YAML::Node calib = YAML::LoadFile(CALIB_YAML);

  Eigen::Matrix3d K;
  for (int i = 0; i < 9; ++i)
    K(i / 3, i % 3) = calib["K"][i].as<double>();

  Eigen::Matrix4d T_P_C = load_T_4x4(CALIB_YAML);
  Eigen::Matrix4d T_cam_lidar = T_P_C.inverse();

  Eigen::Matrix4d T_pitch = Eigen::Matrix4d::Identity();
  T_pitch.block<3,3>(0,0) =
    geometry::Geometry3D::GetRotationMatrixFromXYZ(
      Eigen::Vector3d(-M_PI / 6.0, 0, 0));

  // ----------------------------------------------------------
  // Load templates
  // ----------------------------------------------------------
  auto templates = load_templates(TEMPLATE_DIR);
  if (templates.empty())
    throw std::runtime_error("No templates loaded");

  // ----------------------------------------------------------
  // Scan data directory
  // ----------------------------------------------------------
  std::unordered_map<std::string, fs::path> plys, pngs;

  for (const auto & e : fs::directory_iterator(DATA_DIR)) {
    if (!e.is_regular_file())
      continue;

    auto ext = e.path().extension().string();
    auto ts  = extract_timestamp(e.path());
    if (ts.empty())
      continue;

    if (ext == ".ply")
      plys[ts] = e.path();
    else if (ext == ".png")
      pngs[ts] = e.path();
  }

  std::cout << "Found " << plys.size()
            << " PLYs and " << pngs.size()
            << " PNGs\n";

  // ----------------------------------------------------------
  // Main loop
  // ----------------------------------------------------------
  int scene_idx = 0;
  for (const auto & [ts, ply_path] : plys) {

    if (!pngs.count(ts))
      continue;

    const auto & mask_path = pngs[ts];

    std::cout << "\n========================================\n";
    std::cout << "scene: " << scene_idx << "\n";
    std::cout << "Timestamp: " << ts << "\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    // --------------------------------------------------------
    // Load mask
    // --------------------------------------------------------
    cv::Mat mask =
      cv::imread(mask_path.string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
      std::cerr << "Failed to load mask\n";
      continue;
    }

    // --------------------------------------------------------
    // Load point cloud
    // --------------------------------------------------------
    auto pcd_raw =
      std::make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(ply_path.string(), *pcd_raw) ||
        pcd_raw->IsEmpty()) {
      std::cerr << "Failed to load point cloud\n";
      continue;
    }

    // --------------------------------------------------------
    // Mask-based cutout
    // --------------------------------------------------------
    auto pts_sel =
      select_points_by_mask(
        pcd_raw->points_, mask, K, T_P_C);

    if (pts_sel.empty()) {
      std::cerr << "Empty cutout\n";
      continue;
    }

    auto pcd_cutout =
      std::make_shared<geometry::PointCloud>();
    pcd_cutout->points_ = pts_sel;

    io::WritePointCloud(
      OUT_DIR + "/cutout_" + ts + ".ply", *pcd_cutout);

    // --------------------------------------------------------
    // Preprocess
    // --------------------------------------------------------
    pcd_cutout->RemoveStatisticalOutliers(20, 2.0);
    pcd_cutout->Transform(T_pitch * T_cam_lidar);
    pcd_cutout->RemoveStatisticalOutliers(20, 2.0);
    pcd_cutout->EstimateNormals();

    // --------------------------------------------------------
    // Global registration
    // --------------------------------------------------------
    GlobalRegistrationResult globreg =
      compute_global_registration(
        *pcd_cutout,
        Z_WORLD,
        ANGLE_THRESH,
        MAX_PLANES,
        DIST_THRESH,
        MIN_INLIERS);

    if (!globreg.success)
      std::cerr << "Global registration failed\n";

    std::cout << "nr. of planes: " << globreg.num_planes << "\n";

    // --------------------------------------------------------
    // GLOBAL-ONLY MODE
    // --------------------------------------------------------
    if (MODE == PipelineMode::GLOBAL_ONLY) {

      visualize_global(
        pcd_cutout,
        globreg,
        "Global registration – " + ts);

      continue;
    }

    // --------------------------------------------------------
    // FULL PIPELINE
    // --------------------------------------------------------
    const auto & icp_scene =
      globreg.plane_cloud ? globreg.plane_cloud : pcd_cutout;

    LocalRegistrationResult result =
      compute_local_registration(
        *icp_scene,
        templates,
        globreg,
        ICP_DIST,
        30);

    auto t_end = std::chrono::high_resolution_clock::now();

    // --------------------------------------------------------
    // Report
    // --------------------------------------------------------
    std::cout << "Template : " << result.template_name << "\n";
    std::cout << "Yaw [deg]: " << result.yaw_deg << "\n";
    std::cout << "Fitness  : " << result.icp.fitness_ << "\n";
    std::cout << "RMSE     : " << result.icp.inlier_rmse_ << "\n";
    std::cout << "Time [ms]: "
              << std::chrono::duration<double, std::milli>(
                   t_end - t_start).count()
              << "\n";

    // --------------------------------------------------------
    // Visualization
    // --------------------------------------------------------
    visualize_full(
      icp_scene,
      templates[result.template_index],
      result,
      "Full pipeline – " + ts);
  }

  return 0;
}
