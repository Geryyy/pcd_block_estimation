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
// Data structures
// ------------------------------------------------------------
struct TemplateData
{
  std::string name;
  std::shared_ptr<open3d::geometry::PointCloud> pcd;
  int num_faces;
};

// ------------------------------------------------------------
// Face definitions
// ------------------------------------------------------------
extern const std::map<std::string, Eigen::Vector3d> FACE_NORMALS;
extern const std::map<std::string, std::vector<std::string>> TEMPLATES;

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
// Core functionality
// ------------------------------------------------------------
std::shared_ptr<open3d::geometry::PointCloud>
sample_pointcloud_from_mesh(
  const std::string & mesh_path,
  int n_points);

void ensure_outward_normals(
  open3d::geometry::PointCloud & pcd);

bool normal_matches_faces(
  const Eigen::Vector3d & n,
  const std::vector<std::string> & faces,
  double cos_thresh);

std::shared_ptr<open3d::geometry::PointCloud>
extract_template(
  const open3d::geometry::PointCloud & pcd,
  const std::vector<std::string> & faces,
  double angle_deg);

void write_template(
  const std::string & out_dir,
  const std::string & name,
  const open3d::geometry::PointCloud & pcd,
  const std::vector<std::string> & faces);

void generate_templates(
  const TemplateGenerationParams & params);

// ------------------------------------------------------------
// Existing API (unchanged)
// ------------------------------------------------------------
std::vector<TemplateData>
load_templates(const std::string & directory);

} // namespace pcd_block
