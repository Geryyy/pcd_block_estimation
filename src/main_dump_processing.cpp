#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <unordered_map>
#include <iostream>
#include <sstream>

// ------------------------------------------------------------
// library headers
// ------------------------------------------------------------
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

static constexpr PipelineMode MODE = PipelineMode::FULL_PIPELINE;
// static constexpr PipelineMode MODE = PipelineMode::GLOBAL_ONLY;

// ============================================================
// Paths
// ============================================================
static const std::string DATA_DIR     = "../data/dump";
static const std::string CALIB_YAML   = "../data/calib_zed2i_to_seyond.yaml";
static const std::string TEMPLATE_DIR = "../data/templates";

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
constexpr double MAX_PLANE_CENTER_DIST = 0.6; // meters

// ============================================================
// Forward declarations
// ============================================================
static void visualize_global(
  const std::shared_ptr<geometry::PointCloud> & cutout,
  const GlobalRegistrationResult & glob,
  const std::string & title);

static void visualize_icp_result(
  const std::shared_ptr<geometry::PointCloud> & scene,
  const TemplateData & tpl,
  const LocalRegistrationResult & result,
  const std::string & title);

// ============================================================
// Helpers
// ============================================================
static std::string extract_timestamp(const fs::path & p)
{
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

// ============================================================
// Visualization helpers
// ============================================================
static Eigen::Vector3d plane_color(int i)
{
  static const std::vector<Eigen::Vector3d> colors = {
    {1,0,0},{0,1,0},{0,0,1},{1,1,0},{1,0,1},{0,1,1}
  };
  return colors[i % colors.size()];
}

static std::shared_ptr<geometry::TriangleMesh>
make_normal_cylinder(
  const Eigen::Vector3d & origin,
  const Eigen::Vector3d & normal,
  const Eigen::Vector3d & color)
{
  auto cyl =
    geometry::TriangleMesh::CreateCylinder(0.01, 0.3);
  cyl->PaintUniformColor(color);

  Eigen::Vector3d n = normal.normalized();
  Eigen::Vector3d z(0,0,1);

  Eigen::Matrix3d R =
    Eigen::Quaterniond::FromTwoVectors(z, n).toRotationMatrix();

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3,3>(0,0) = R;
  T.block<3,1>(0,3) = origin + 0.15 * n;

  cyl->Transform(T);
  cyl->ComputeVertexNormals();
  return cyl;
}

// ------------------------------------------------------------
// GLOBAL visualization
// ------------------------------------------------------------
static void visualize_global(
  const std::shared_ptr<geometry::PointCloud> & cutout,
  const GlobalRegistrationResult & glob,
  const std::string & title)
{
  std::vector<std::shared_ptr<const geometry::Geometry>> geoms;

  auto bg = std::make_shared<geometry::PointCloud>(*cutout);
  bg->PaintUniformColor({0.6,0.6,0.6});
  geoms.push_back(bg);

  for (size_t i = 0; i < glob.planes.size(); ++i) {
    const auto & pc = glob.planes[i].second;
    Eigen::Vector3d color = plane_color(i);

    auto pc_vis =
      std::make_shared<geometry::PointCloud>(pc);
    pc_vis->PaintUniformColor(color);
    geoms.push_back(pc_vis);

    Eigen::Vector3d c = compute_center(pc.points_);
    Eigen::Vector3d n =
      glob.planes[i].first.head<3>().normalized();

    if (n.dot(Z_WORLD) < 0) n = -n;

    geoms.push_back(make_normal_cylinder(c, n, color));
  }

  visualization::DrawGeometries(geoms, title, 1200, 900);
}

// ------------------------------------------------------------
// ICP visualization
// ------------------------------------------------------------
static void visualize_icp_result(
  const std::shared_ptr<geometry::PointCloud> & scene,
  const TemplateData & tpl,
  const LocalRegistrationResult & result,
  const std::string & title)
{
  std::vector<std::shared_ptr<const geometry::Geometry>> geoms;

  auto scene_vis =
    std::make_shared<geometry::PointCloud>(*scene);
  scene_vis->PaintUniformColor({0,0.8,0});
  geoms.push_back(scene_vis);

  auto tpl_vis =
    std::make_shared<geometry::PointCloud>(*tpl.pcd);
  tpl_vis->Transform(result.icp.transformation_);
  tpl_vis->PaintUniformColor({1,0,0});
  geoms.push_back(tpl_vis);

  auto frame =
    geometry::TriangleMesh::CreateCoordinateFrame(0.25);
  frame->Transform(result.icp.transformation_);
  geoms.push_back(frame);

  visualization::DrawGeometries(geoms, title, 1200, 900);
}

// ============================================================
// MAIN
// ============================================================
int main()
{
  // ----------------------------------------------------------
  // Load calibration
  // ----------------------------------------------------------
  YAML::Node calib = YAML::LoadFile(CALIB_YAML);

  Eigen::Matrix3d K;
  for (int i = 0; i < 9; ++i)
    K(i/3, i%3) = calib["K"][i].as<double>();

  Eigen::Matrix4d T_P_C = load_T_4x4(CALIB_YAML);
  Eigen::Matrix4d T_cam_lidar = T_P_C.inverse();

  Eigen::Matrix4d T_pitch = Eigen::Matrix4d::Identity();
  T_pitch.block<3,3>(0,0) =
    geometry::Geometry3D::GetRotationMatrixFromXYZ(
      Eigen::Vector3d(-M_PI/6,0,0));

  // ----------------------------------------------------------
  // Load templates
  // ----------------------------------------------------------
  auto templates = load_templates(TEMPLATE_DIR);
  if (templates.empty())
    throw std::runtime_error("No templates loaded");

  // ----------------------------------------------------------
  // Scan dump directory
  // ----------------------------------------------------------
  std::unordered_map<std::string, fs::path> plys, pngs;

  for (const auto & e : fs::directory_iterator(DATA_DIR)) {
    if (!e.is_regular_file()) continue;
    auto ts = extract_timestamp(e.path());
    if (ts.empty()) continue;
    if (e.path().extension() == ".ply") plys[ts] = e.path();
    if (e.path().extension() == ".png") pngs[ts] = e.path();
  }

  std::cout << "[INFO] Found "
            << plys.size() << " PLYs and "
            << pngs.size() << " PNGs\n";

  // ----------------------------------------------------------
  // Main loop
  // ----------------------------------------------------------
  for (const auto & [ts, ply_path] : plys) {

    if (!pngs.count(ts)) continue;

    std::cout << "\n==============================\n";
    std::cout << "[SCENE] timestamp = " << ts << "\n";

    cv::Mat mask =
      cv::imread(pngs[ts].string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty()) continue;

    auto pcd_raw = std::make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(ply_path.string(), *pcd_raw))
      continue;

    auto pts_sel =
      select_points_by_mask(
        pcd_raw->points_, mask, K, T_P_C);

    if (pts_sel.empty()) continue;

    auto pcd_cutout =
      std::make_shared<geometry::PointCloud>();
    pcd_cutout->points_ = pts_sel;

    pcd_cutout->RemoveStatisticalOutliers(20,2.0);
    pcd_cutout->Transform(T_pitch * T_cam_lidar);
    pcd_cutout->RemoveStatisticalOutliers(20,2.0);
    pcd_cutout->EstimateNormals();

    // --------------------------------------------------------
    // GLOBAL REGISTRATION
    // --------------------------------------------------------
    auto glob =
      compute_global_registration(
        *pcd_cutout,
        Z_WORLD,
        ANGLE_THRESH,
        MAX_PLANES,
        DIST_THRESH,
        MIN_INLIERS,
    MAX_PLANE_CENTER_DIST);

    if (!glob.success) {
      std::cout << "[GLOBAL] ❌ failed\n";
      continue;
    }

    std::cout << "[GLOBAL] num_planes = "
              << glob.num_planes << "\n";

    for (size_t i = 0; i < glob.planes.size(); ++i) {
      Eigen::Vector3d n =
        glob.planes[i].first.head<3>().normalized();
      double cosz = std::abs(n.dot(Z_WORLD));
      std::cout << "  plane[" << i << "] "
                << "n=" << n.transpose()
                << " cos(z)=" << cosz
                << " inliers="
                << glob.planes[i].second.points_.size()
                << "\n";
    }

    std::cout << "[GLOBAL] top normal = "
              << glob.n_top.transpose() << "\n";

    for (size_t i = 0; i < glob.front_normals.size(); ++i) {
      std::cout << "[GLOBAL] front[" << i << "] n="
                << glob.front_normals[i].transpose()
                << " c="
                << glob.front_centers[i].transpose()
                << "\n";
    }

    if (MODE == PipelineMode::GLOBAL_ONLY) {
      visualize_global(
        pcd_cutout,
        glob,
        "Global registration – " + ts);
      continue;
    }

    // --------------------------------------------------------
    // LOCAL REGISTRATION
    // --------------------------------------------------------
    const auto & icp_scene =
      glob.plane_cloud ? glob.plane_cloud : pcd_cutout;

    auto result =
      compute_local_registration(
        *icp_scene,
        templates,
        glob,
        ICP_DIST);

    if (!result.success) {
      std::cout << "[RESULT] ❌ no valid ICP result\n";
      continue;
    }

    std::cout << "[RESULT] template = "
              << result.template_name << "\n";
    std::cout << "         front_plane = "
              << result.front_plane_index << "\n";
    std::cout << "         face = "
              << (result.face_index==0?"LONG":"SHORT") << "\n";
    std::cout << "         fitness = "
              << result.icp.fitness_ << "\n";
    std::cout << "         rmse = "
              << result.icp.inlier_rmse_ << "\n";

    visualize_icp_result(
      icp_scene,
      templates[result.template_index],
      result,
      "ICP result – " + ts);
  }

  return 0;
}
