#include "pcd_block_estimation/pose_estimation.hpp"
#include "pcd_block_estimation/utils.hpp"

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>

using namespace open3d;

namespace pcd_block
{

// ============================================================
// Helpers
// ============================================================

static Eigen::Vector3d
plane_normal_oriented(
  const Eigen::Vector4d & plane,
  const Eigen::Vector3d & z_world)
{
  Eigen::Vector3d n = plane.head<3>().normalized();
  if (n.dot(z_world) < 0.0)
    n = -n;
  return n;
}


// ============================================================
// GLOBAL REGISTRATION
// ============================================================

GlobalRegistrationResult compute_global_registration(
  const geometry::PointCloud & scene,
  const Eigen::Vector3d & z_world,
  double angle_thresh,
  int max_planes,
  double dist_thresh,
  int min_inliers,
  double max_plane_center_dist)
{
  GlobalRegistrationResult out;

  // ----------------------------------------------------------
  // Scene centroid
  // ----------------------------------------------------------
  out.center = compute_center(scene);

  // ----------------------------------------------------------
  // Plane extraction
  // ----------------------------------------------------------
  out.planes =
    extract_planes(scene, max_planes, dist_thresh, min_inliers);

  if (out.planes.empty()) {
    std::cerr << "[globreg] no planes detected\n";
    return out;
  }

  out.num_planes = static_cast<int>(out.planes.size());

  // Concatenate plane inliers
  out.plane_cloud = std::make_shared<geometry::PointCloud>();
  for (const auto & [_, pc] : out.planes)
    *out.plane_cloud += pc;

  // ----------------------------------------------------------
  // Identify top + front candidates
  // ----------------------------------------------------------
  bool found_top = false;

  struct FrontCandidate {
    Eigen::Vector3d n;
    Eigen::Vector3d c;
    size_t support;
  };

  std::vector<FrontCandidate> front_candidates;

  for (const auto & [plane, pc] : out.planes) {

    Eigen::Vector3d n =
      plane_normal_oriented(plane, z_world);
    Eigen::Vector3d c = compute_center(pc);

    double cos_to_z = std::abs(n.dot(z_world));

    // ----------------------------
    // Top plane
    // ----------------------------
    if (!found_top && cos_to_z > angle_thresh) {
      out.n_top = n;
      out.c_top = c;
      found_top = true;
      continue;
    }

    // ----------------------------
    // Vertical planes → front candidates
    // ----------------------------
    if (cos_to_z < 0.2) {
      front_candidates.push_back({
        n, c, pc.points_.size()
      });
    }
  }

  if (!found_top) {
    std::cerr << "[globreg] top plane not found\n";
    return out;
  }

  if (front_candidates.empty()) {
    std::cerr << "[globreg] no vertical planes found\n";
    return out;
  }

  // ----------------------------------------------------------
  // Reject parallel vertical planes
  // (ground leakage often produces this)
  // ----------------------------------------------------------
  for (size_t i = 0; i < front_candidates.size(); ++i) {
    for (size_t j = i + 1; j < front_candidates.size(); ++j) {

      double cos_ang =
        std::abs(front_candidates[i].n.dot(
                 front_candidates[j].n));

      if (cos_ang > 0.95) {
        std::cerr
          << "[globreg] rejecting: parallel vertical planes "
          << "(cos=" << cos_ang << ")\n";
        return out;
      }
    }
  }

  // ----------------------------------------------------------
  // Reject planes too far apart (mask leakage / ground)
  // ----------------------------------------------------------
  for (const auto & fc : front_candidates) {
    double d = (fc.c - out.c_top).norm();
    if (d > max_plane_center_dist) {
      std::cerr
        << "[globreg] rejecting: plane too far from top ("
        << d << " m)\n";
      return out;
    }
  }

  // ----------------------------------------------------------
  // Sort front planes by support
  // ----------------------------------------------------------
  std::sort(
    front_candidates.begin(),
    front_candidates.end(),
    [](const auto & a, const auto & b) {
      return a.support > b.support;
    });

  // Keep at most 2 (45° ambiguity case)
  constexpr size_t MAX_FRONT_PLANES = 2;
  for (size_t i = 0;
       i < std::min(front_candidates.size(),
                    MAX_FRONT_PLANES);
       ++i) {

    out.front_normals.push_back(front_candidates[i].n);
    out.front_centers.push_back(front_candidates[i].c);
  }

  // ----------------------------------------------------------
  // Build yaw-free base frame from top plane
  // ----------------------------------------------------------
  Eigen::Vector3d z = out.n_top;

  Eigen::Vector3d tmp(1.0, 0.0, 0.0);
  if (std::abs(tmp.dot(z)) > 0.9)
    tmp = Eigen::Vector3d(0.0, 1.0, 0.0);

  Eigen::Vector3d x =
    (tmp - tmp.dot(z) * z).normalized();
  Eigen::Vector3d y = z.cross(x).normalized();

  out.R_base.col(0) = x;
  out.R_base.col(1) = y;
  out.R_base.col(2) = z;

  out.success = true;
  return out;
}


// ============================================================
// LOCAL REGISTRATION (plane snapping + ICP)
// ============================================================

LocalRegistrationResult compute_local_registration(
  const geometry::PointCloud & scene,
  const std::vector<TemplateData> & templates,
  const GlobalRegistrationResult & glob,
  double icp_dist)
{
  LocalRegistrationResult best;
  best.icp.fitness_ = -1.0;

  if (!glob.success)
    return best;

  for (size_t ti = 0; ti < templates.size(); ++ti) {
    const auto & tpl = templates[ti];

    Eigen::Vector3d tpl_center =
      compute_center(*tpl.pcd);

    for (size_t fi = 0; fi < glob.front_normals.size(); ++fi) {

      const Eigen::Vector3d & n_front = glob.front_normals[fi];
      const Eigen::Vector3d & c_top   = glob.c_top;

      for (int face = 0; face < 2; ++face) {

        const Eigen::Vector3d & tpl_side_n =
          (face == 0) ? tpl.normal_long : tpl.normal_short;

        Eigen::Matrix3d R_top =
          Eigen::Quaterniond::FromTwoVectors(
            tpl.normal_top,
            glob.n_top).toRotationMatrix();

        Eigen::Vector3d side_rot = R_top * tpl_side_n;

        Eigen::Matrix3d R_yaw =
          Eigen::Quaterniond::FromTwoVectors(
            side_rot,
            n_front).toRotationMatrix();

        Eigen::Matrix3d R = R_yaw * R_top;

        Eigen::Vector3d t =
          c_top - R * tpl_center;

        Eigen::Matrix4d T_init = Eigen::Matrix4d::Identity();
        T_init.block<3,3>(0,0) = R;
        T_init.block<3,1>(0,3) = t;

        auto icp =
          pipelines::registration::RegistrationICP(
            *tpl.pcd,
            scene,
            icp_dist,
            T_init,
            pipelines::registration::
              TransformationEstimationPointToPlane());

        if (icp.fitness_ > best.icp.fitness_) {
          best.success = true;
          best.template_name = tpl.name;
          best.template_index = static_cast<int>(ti);
          best.front_plane_index = static_cast<int>(fi);
          best.face_index = face;
          best.icp = icp;
        }
      }
    }
  }

  return best;
}

// ============================================================
// Utility
// ============================================================

Eigen::Matrix4d
globalResultToTransform(const GlobalRegistrationResult & glob)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

  if (!glob.success)
    return T;

  T.block<3,3>(0,0) = glob.R_base;
  T.block<3,1>(0,3) = glob.center;

  return T;
}

} // namespace pcd_block
