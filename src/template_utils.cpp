#include "pcd_block_estimation/template_utils.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>

namespace fs = std::filesystem;
using namespace open3d;

namespace pcd_block
{

// ------------------------------------------------------------
// Canonical face normals (CAD frame)
// ------------------------------------------------------------
const std::map<std::string, Eigen::Vector3d> FACE_NORMALS = {
  {"top",   {0.0, 0.0, 1.0}},
  {"front", {1.0, 0.0, 0.0}},  // short side
  {"right", {0.0, 1.0, 0.0}},  // long side
  {"left",  {0.0,-1.0, 0.0}}
};

// ------------------------------------------------------------
// Template definitions
// ------------------------------------------------------------
const std::map<std::string, std::vector<std::string>> TEMPLATES = {
  {"top", {"top"}},

  {"top_front_short", {"top", "front"}},
  {"top_front_long",  {"top", "right"}},

  {"top_front_short_left",  {"top", "front", "left"}},
  {"top_front_short_right", {"top", "front", "right"}},
};

// ------------------------------------------------------------
std::shared_ptr<geometry::PointCloud>
sample_pointcloud_from_mesh(
  const std::string & mesh_path,
  int n_points)
{
  auto mesh = std::make_shared<geometry::TriangleMesh>();
  if (!io::ReadTriangleMesh(mesh_path, *mesh) || mesh->IsEmpty()) {
    throw std::runtime_error("Failed to load mesh: " + mesh_path);
  }

  mesh->ComputeVertexNormals();
  auto pcd = mesh->SamplePointsPoissonDisk(n_points);

  if (!pcd || pcd->IsEmpty()) {
    throw std::runtime_error("Poisson disk sampling failed");
  }

  return pcd;
}

// ------------------------------------------------------------
void ensure_outward_normals(geometry::PointCloud & pcd)
{
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (const auto & p : pcd.points_)
    center += p;
  center /= static_cast<double>(pcd.points_.size());

  for (size_t i = 0; i < pcd.normals_.size(); ++i) {
    if (pcd.normals_[i].dot(pcd.points_[i] - center) < 0.0)
      pcd.normals_[i] *= -1.0;
  }
}

// ------------------------------------------------------------
static bool normal_matches_faces(
  const Eigen::Vector3d & n,
  const std::vector<std::string> & faces,
  double cos_thresh)
{
  for (const auto & f : faces) {
    const auto & fn = FACE_NORMALS.at(f);
    if (n.dot(fn) > cos_thresh)
      return true;
  }
  return false;
}

// ------------------------------------------------------------
std::shared_ptr<geometry::PointCloud>
extract_template(
  const geometry::PointCloud & pcd,
  const std::vector<std::string> & faces,
  double angle_deg)
{
  auto out = std::make_shared<geometry::PointCloud>();
  const double cos_thresh = std::cos(angle_deg * M_PI / 180.0);

  for (size_t i = 0; i < pcd.points_.size(); ++i) {
    if (normal_matches_faces(pcd.normals_[i], faces, cos_thresh)) {
      out->points_.push_back(pcd.points_[i]);
      out->normals_.push_back(pcd.normals_[i]);
    }
  }

  return out;
}

// ------------------------------------------------------------
void write_template(
  const std::string & out_dir,
  const std::string & name,
  const geometry::PointCloud & pcd,
  const std::vector<std::string> & faces)
{
  fs::create_directories(out_dir);

  io::WritePointCloud(out_dir + "/" + name + ".ply", pcd, false);

  YAML::Node meta;
  meta["template_name"] = name;
  meta["num_faces"] = static_cast<int>(faces.size());

  YAML::Node faces_node;
  for (const auto & f : faces) {
    YAML::Node fn;
    fn["name"] = f;

    const auto & n = FACE_NORMALS.at(f);
    fn["normal"] = std::vector<double>{n.x(), n.y(), n.z()};
    faces_node.push_back(fn);
  }

  meta["faces"] = faces_node;

  std::ofstream fout(out_dir + "/" + name + ".yaml");
  fout << meta;
}

// ------------------------------------------------------------
void generate_templates(const TemplateGenerationParams & params)
{
  auto pcd = sample_pointcloud_from_mesh(
    params.cad_path, params.n_points);

  ensure_outward_normals(*pcd);

  for (const auto &[name, faces] : TEMPLATES) {

    auto tpl = extract_template(
      *pcd, faces, params.angle_deg);

    std::cout << name << ": "
              << tpl->points_.size() << " points\n";

    if (tpl->IsEmpty()) {
      std::cerr << "WARNING: empty template " << name << "\n";
      continue;
    }

    write_template(params.out_dir, name, *tpl, faces);
  }
}

// ------------------------------------------------------------
// Runtime loader
// ------------------------------------------------------------
std::vector<TemplateData>
load_templates(const std::string & dir)
{
  std::vector<TemplateData> out;

  for (const auto & f : fs::directory_iterator(dir)) {

    if (f.path().extension() != ".ply")
      continue;

    fs::path yaml_path = f.path();
    yaml_path.replace_extension(".yaml");
    if (!fs::exists(yaml_path))
      continue;

    YAML::Node meta = YAML::LoadFile(yaml_path.string());

    auto pcd = std::make_shared<geometry::PointCloud>();
    io::ReadPointCloud(f.path().string(), *pcd);
    pcd->EstimateNormals();

    TemplateData tpl;
    tpl.name = f.path().stem().string();
    tpl.pcd  = pcd;
    tpl.num_faces = meta["num_faces"].as<int>();

    for (const auto & fnode : meta["faces"]) {
      tpl.face_names.push_back(
        fnode["name"].as<std::string>());
    }

    // canonical normals (fixed by CAD)
    tpl.normal_top   = Eigen::Vector3d(0, 0, 1);
    tpl.normal_short = Eigen::Vector3d(1, 0, 0);
    tpl.normal_long  = Eigen::Vector3d(0, 1, 0);

    out.push_back(tpl);
  }

  return out;
}

} // namespace pcd_block
