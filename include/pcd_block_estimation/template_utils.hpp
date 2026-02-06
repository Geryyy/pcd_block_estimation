#pragma once

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace pcd_block
{

// ------------------------------------------------------------
// Canonical face normals (CAD frame)
// ------------------------------------------------------------
extern const std::map<std::string, Eigen::Vector3d> FACE_NORMALS;

// ------------------------------------------------------------
// Template metadata (RUNTIME)
// ------------------------------------------------------------
struct TemplateData
{
  std::string name;
  std::shared_ptr<open3d::geometry::PointCloud> pcd;

  // semantic info
  int num_faces = 0;
  std::vector<std::string> face_names;

  // canonical normals (CAD frame)
  Eigen::Vector3d normal_top;    // +Z
  Eigen::Vector3d normal_short;  // +X
  Eigen::Vector3d normal_long;   // +Y

  // canonical face centers (CAD frame)
  Eigen::Vector3d center_top;
  Eigen::Vector3d center_short;
  Eigen::Vector3d center_long;
};

// ------------------------------------------------------------
// Template generation params
// ------------------------------------------------------------
struct TemplateGenerationParams
{
  std::string cad_path;
  std::string out_dir;
  int n_points = 2000;
  double angle_deg = 15.0;
};

// ------------------------------------------------------------
// Template generation
// ------------------------------------------------------------
std::shared_ptr<open3d::geometry::PointCloud>
sample_pointcloud_from_mesh(
  const std::string & mesh_path,
  int n_points);

void ensure_outward_normals(
  open3d::geometry::PointCloud & pcd);

std::shared_ptr<open3d::geometry::PointCloud>
extract_template(
  const open3d::geometry::PointCloud & pcd,
  const std::vector<std::string> & faces,
  double angle_deg);

void generate_templates(
  const TemplateGenerationParams & params);

// ------------------------------------------------------------
// Loader (runtime)
// ------------------------------------------------------------
std::vector<TemplateData>
load_templates(const std::string & directory);

} // namespace pcd_block
