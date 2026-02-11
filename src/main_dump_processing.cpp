#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <unordered_map>
#include <iostream>
#include <sstream>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

// ------------------------------------------------------------
// library headers
// ------------------------------------------------------------
#include "pcd_block_estimation/yaml_utils.hpp"
#include "pcd_block_estimation/utils.hpp"
#include "pcd_block_estimation/mask_projection.hpp"
#include "pcd_block_estimation/template_utils.hpp"
#include "pcd_block_estimation/pose_estimation.hpp"
#include "pcd_block_estimation/Timer.hpp"

using namespace open3d;
using namespace pcd_block;
namespace fs = std::filesystem;

// ============================================================
// MODE SWITCH
// ============================================================
enum class PipelineMode
{
  GLOBAL_ONLY,
  FULL_PIPELINE
};

static constexpr PipelineMode MODE = PipelineMode::FULL_PIPELINE;
// static constexpr PipelineMode MODE = PipelineMode::GLOBAL_ONLY;

static constexpr bool DEBUG_PIPELINE_VISUALIZATION = true;
constexpr double DEBUG_OFFSET_X = 1.5;
constexpr bool visualize_failed_global = false;
constexpr bool visualize_failed_local = true;

// ============================================================
// Paths
// ============================================================
static const std::string DATA_DIR = "../data/dump";
static const std::string CALIB_YAML = "../data/calib_zed2i_to_seyond.yaml";
static const std::string TEMPLATE_DIR = "../data/templates";

// ============================================================
// Parameters
// ============================================================
constexpr double DIST_THRESH = 0.02;
constexpr int MAX_PLANES = 2;
constexpr int MIN_INLIERS = 100;
constexpr double ICP_DIST = 0.1;

const Eigen::Vector3d Z_WORLD(0.0, -1.0, 0.0);
constexpr double ANGLE_THRESH =
    std::cos(30.0 * M_PI / 180.0);
constexpr double MAX_PLANE_CENTER_DIST = 0.6; // meters

// ============================================================
// Helpers
// ============================================================
static std::string extract_timestamp(const fs::path &p)
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

static Eigen::Matrix4d make_x_offset(double x)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T(0, 3) = x;
  return T;
}

inline void compute_pose_error(
    const Eigen::Matrix4d &H_global,
    const Eigen::Matrix4d &H_local,
    double &translation_error,
    double &rotation_error_rad,
    double &rotation_error_deg)
{
  // --- Translation ---
  const Eigen::Vector3d t_g = H_global.block<3, 1>(0, 3);
  const Eigen::Vector3d t_l = H_local.block<3, 1>(0, 3);

  translation_error = (t_g - t_l).norm();

  // --- Rotation ---
  const Eigen::Matrix3d R_g = H_global.block<3, 3>(0, 0);
  const Eigen::Matrix3d R_l = H_local.block<3, 3>(0, 0);

  const Eigen::Matrix3d R_err = R_g.transpose() * R_l;

  double cos_angle = 0.5 * (R_err.trace() - 1.0);
  cos_angle = std::clamp(cos_angle, -1.0, 1.0);

  rotation_error_rad = std::acos(cos_angle);
  rotation_error_deg = rotation_error_rad * 180.0 / M_PI;
}

// ============================================================
// Visualization helpers
// ============================================================
static Eigen::Vector3d plane_color(int i)
{
  static const std::vector<Eigen::Vector3d> colors = {
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
  return colors[i % colors.size()];
}

static std::shared_ptr<geometry::TriangleMesh>
make_normal_cylinder(
    const Eigen::Vector3d &origin,
    const Eigen::Vector3d &normal,
    const Eigen::Vector3d &color)
{
  auto cyl =
      geometry::TriangleMesh::CreateCylinder(0.01, 0.3);
  cyl->PaintUniformColor(color);

  Eigen::Vector3d n = normal.normalized();
  Eigen::Vector3d z(0, 0, 1);

  Eigen::Matrix3d R =
      Eigen::Quaterniond::FromTwoVectors(z, n).toRotationMatrix();

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = origin + 0.15 * n;

  cyl->Transform(T);
  cyl->ComputeVertexNormals();
  return cyl;
}

// ------------------------------------------------------------
// GLOBAL visualization
// ------------------------------------------------------------
static void visualize_global(
    const std::shared_ptr<geometry::PointCloud> &cutout,
    const GlobalRegistrationResult &glob,
    const std::string &title)
{
  std::vector<std::shared_ptr<const geometry::Geometry>> geoms;

  auto bg = std::make_shared<geometry::PointCloud>(*cutout);
  bg->PaintUniformColor({0.6, 0.6, 0.6});
  geoms.push_back(bg);

  for (size_t i = 0; i < glob.planes.size(); ++i)
  {
    const auto &pc = glob.planes[i].second;
    Eigen::Vector3d color = plane_color(i);

    auto pc_vis =
        std::make_shared<geometry::PointCloud>(pc);
    pc_vis->PaintUniformColor(color);
    geoms.push_back(pc_vis);

    Eigen::Vector3d c = compute_center(pc.points_);
    Eigen::Vector3d n =
        glob.planes[i].first.head<3>().normalized();

    if (n.dot(Z_WORLD) < 0)
      n = -n;

    geoms.push_back(make_normal_cylinder(c, n, color));
  }

  auto glob_transform = globalResultToTransform(glob);
  auto frame =
      geometry::TriangleMesh::CreateCoordinateFrame(0.25);
  frame->Transform(glob_transform);
  geoms.push_back(frame);

  visualization::DrawGeometries(geoms, title, 1200, 900);
}

// ------------------------------------------------------------
// ICP visualization
// ------------------------------------------------------------
static void visualize_icp_result(
    const std::shared_ptr<geometry::PointCloud> &scene,
    const TemplateData &tpl,
    const LocalRegistrationResult &result,
    const std::string &title)
{
  std::vector<std::shared_ptr<const geometry::Geometry>> geoms;

  auto scene_vis =
      std::make_shared<geometry::PointCloud>(*scene);
  scene_vis->PaintUniformColor({0, 0.8, 0});
  geoms.push_back(scene_vis);

  auto tpl_vis =
      std::make_shared<geometry::PointCloud>(*tpl.pcd);
  tpl_vis->Transform(result.icp.transformation_);
  tpl_vis->PaintUniformColor({1, 0, 0});
  geoms.push_back(tpl_vis);

  auto frame =
      geometry::TriangleMesh::CreateCoordinateFrame(0.25);
  frame->Transform(result.icp.transformation_);
  geoms.push_back(frame);

  visualization::DrawGeometries(geoms, title, 1200, 900);
}

static void visualize_debug_pipeline(
    const std::shared_ptr<geometry::PointCloud> &original,
    const Eigen::Matrix4d &H_global,
    const std::shared_ptr<geometry::PointCloud> &icp_scene,
    const TemplateData &tpl,
    const LocalRegistrationResult &local,
    const std::string &title)
{
  std::vector<std::shared_ptr<const geometry::Geometry>> geoms;

  // ---------------- ORIGINAL ----------------
  {
    auto pc = std::make_shared<geometry::PointCloud>(*original);
    pc->PaintUniformColor({0.6, 0.6, 0.6});
    geoms.push_back(pc);

    auto frame = geometry::TriangleMesh::CreateCoordinateFrame(0.25);
    geoms.push_back(frame);
  }

  // ---------------- GLOBAL ----------------
  {
    Eigen::Matrix4d T = make_x_offset(DEBUG_OFFSET_X);

    auto pc = std::make_shared<geometry::PointCloud>(*icp_scene);
    pc->PaintUniformColor({0, 0.6, 1});
    pc->Transform(T);
    geoms.push_back(pc);

    auto frame = geometry::TriangleMesh::CreateCoordinateFrame(0.25);
    frame->Transform(T * H_global);
    geoms.push_back(frame);
  }

  // ---------------- LOCAL ----------------
  {
    Eigen::Matrix4d T = make_x_offset(2.0 * DEBUG_OFFSET_X);

    auto scene_vis = std::make_shared<geometry::PointCloud>(*icp_scene);
    scene_vis->PaintUniformColor({0, 0.8, 0});
    scene_vis->Transform(T);
    geoms.push_back(scene_vis);

    auto tpl_vis = std::make_shared<geometry::PointCloud>(*tpl.pcd);
    tpl_vis->Transform(local.icp.transformation_);
    tpl_vis->Transform(T);
    tpl_vis->PaintUniformColor({1, 0, 0});
    geoms.push_back(tpl_vis);

    auto frame = geometry::TriangleMesh::CreateCoordinateFrame(0.25);
    frame->Transform(T * local.icp.transformation_);
    geoms.push_back(frame);
  }

  visualization::DrawGeometries(geoms, title, 1600, 900);
}

static void visualize_original_only(
    const std::shared_ptr<geometry::PointCloud> &original,
    const std::string &title)
{
  auto pc = std::make_shared<geometry::PointCloud>(*original);
  pc->PaintUniformColor({0.7, 0.7, 0.7});

  auto frame =
      geometry::TriangleMesh::CreateCoordinateFrame(0.25);

  visualization::DrawGeometries({pc, frame}, title, 1200, 900);
}

// ============================================================
// MAIN
// ============================================================
int main()
{
  Timer exec_timer;
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
  T_pitch.block<3, 3>(0, 0) =
      geometry::Geometry3D::GetRotationMatrixFromXYZ(
          Eigen::Vector3d(-M_PI / 6, 0, 0));

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

  for (const auto &e : fs::directory_iterator(DATA_DIR))
  {
    if (!e.is_regular_file())
      continue;
    auto ts = extract_timestamp(e.path());
    if (ts.empty())
      continue;
    if (e.path().extension() == ".ply")
      plys[ts] = e.path();
    if (e.path().extension() == ".png")
      pngs[ts] = e.path();
  }

  std::cout << "[INFO] Found "
            << plys.size() << " PLYs and "
            << pngs.size() << " PNGs\n";

  // ----------------------------------------------------------
  // Main loop
  // ----------------------------------------------------------
  for (const auto &[ts, ply_path] : plys)
  {

    if (!pngs.count(ts))
      continue;

    std::cout << "\n==============================\n";
    std::cout << "[SCENE] timestamp = " << ts << "\n";

    cv::Mat mask =
        cv::imread(pngs[ts].string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty())
      continue;

    auto pcd_raw = std::make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(ply_path.string(), *pcd_raw))
      continue;

    auto pts_sel =
        select_points_by_mask(
            pcd_raw->points_, mask, K, T_P_C);

    if (pts_sel.empty())
      continue;

    auto pcd_cutout =
        std::make_shared<geometry::PointCloud>();
    pcd_cutout->points_ = pts_sel;

    pcd_cutout->RemoveStatisticalOutliers(50, 2.0);
    pcd_cutout->Transform(T_pitch * T_cam_lidar);
    // pcd_cutout->RemoveStatisticalOutliers(20,2.0);
    pcd_cutout->EstimateNormals();

    // --------------------------------------------------------
    // GLOBAL REGISTRATION
    // --------------------------------------------------------
    exec_timer.tic();
    auto glob =
        compute_global_registration(
            *pcd_cutout,
            Z_WORLD,
            ANGLE_THRESH,
            MAX_PLANES,
            DIST_THRESH,
            MIN_INLIERS,
            MAX_PLANE_CENTER_DIST);

    std::cout << "[GLOBAL] Exec.Time: " << exec_timer.toc() / 1e9 << std::endl;

    if (!glob.success)
    {
      std::cout << "[GLOBAL] ❌ failed\n";

      if (visualize_failed_global)
      {
        visualize_original_only(
            pcd_cutout,
            "GLOBAL FAILED – " + ts);
      }

      continue;
    }

    std::cout << "[GLOBAL] num_planes = "
              << glob.num_planes << "\n";

    for (size_t i = 0; i < glob.planes.size(); ++i)
    {
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

    for (size_t i = 0; i < glob.front_normals.size(); ++i)
    {
      std::cout << "[GLOBAL] front[" << i << "] n="
                << glob.front_normals[i].transpose()
                << " c="
                << glob.front_centers[i].transpose()
                << "\n";
    }

    const auto &icp_scene =
        glob.plane_cloud ? glob.plane_cloud : pcd_cutout;

    icp_scene->RemoveStatisticalOutliers(50, 2.0);

    if (MODE == PipelineMode::GLOBAL_ONLY)
    {
      visualize_global(
          icp_scene,
          glob,
          "Global registration – " + ts);
      continue;
    }

    auto H_global = globalResultToTransform(glob);

    // --------------------------------------------------------
    // LOCAL REGISTRATION
    // --------------------------------------------------------

    exec_timer.tic();

    auto result =
        compute_local_registration(
            *icp_scene,
            templates,
            glob,
            ICP_DIST);

    std::cout << "[LOCAL] Exec.Time: " << exec_timer.toc() / 1e9 << std::endl;

    if (!result.success)
    {
      std::cout << "[RESULT] ❌ no valid ICP result\n";

      if (visualize_failed_local)
      {
        // show original + global side by side
        std::vector<std::shared_ptr<const geometry::Geometry>> geoms;

        // ORIGINAL
        {
          auto pc = std::make_shared<geometry::PointCloud>(*pcd_cutout);
          pc->PaintUniformColor({0.6, 0.6, 0.6});
          geoms.push_back(pc);
        }

        // GLOBAL
        {
          Eigen::Matrix4d T = make_x_offset(DEBUG_OFFSET_X);

          auto pc = std::make_shared<geometry::PointCloud>(*icp_scene);
          pc->PaintUniformColor({0, 0.6, 1});
          pc->Transform(T);
          geoms.push_back(pc);

          auto frame =
              geometry::TriangleMesh::CreateCoordinateFrame(0.25);
          frame->Transform(T * H_global);
          geoms.push_back(frame);
        }

        visualization::DrawGeometries(
            geoms,
            "LOCAL FAILED – " + ts,
            1400,
            900);
      }

      continue;
    }
    std::cout << "[RESULT] template[" << result.template_index << "]= "
              << result.template_name << "\n";
    // std::cout << "         front_plane = "
    // << result.front_plane_index << "\n";
    // std::cout << "         face = "
    // << (result.face_index==0?"LONG":"SHORT") << "\n";
    std::cout << "         fitness = "
              << result.icp.fitness_ << "\n";
    std::cout << "         rmse = "
              << result.icp.inlier_rmse_ << "\n";

    auto H_local = result.icp.transformation_;

    // visualize_icp_result(
    //     icp_scene,
    //     templates[result.template_index],
    //     result,
    //     "ICP result – " + ts);

    // error computation: global vs local
    double t_err, r_err_rad, r_err_deg;

    compute_pose_error(
        H_global,
        H_local,
        t_err,
        r_err_rad,
        r_err_deg);

    std::cout
        << "translation error: " << t_err << " m\n"
        << "rotation error:    " << r_err_deg << " deg\n";

    if (DEBUG_PIPELINE_VISUALIZATION)
    {
      visualize_debug_pipeline(
          pcd_cutout,
          H_global,
          icp_scene,
          templates[result.template_index],
          result,
          "DEBUG pipeline – " + ts);
    }
  }

  return 0;
}
