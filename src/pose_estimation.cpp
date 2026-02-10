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
  orient_normal_towards(
      const Eigen::Vector4d &plane,
      const Eigen::Vector3d &reference_dir)
  {
    Eigen::Vector3d n = plane.head<3>().normalized();

    if (n.dot(reference_dir) < 0.0)
    {
      n = -n;
    }
    return n;
  }

  static bool
  is_front_plane_square_pca(
      const geometry::PointCloud &pc_front,
      const Eigen::Vector3d &c_front,
      const Eigen::Vector3d &n_front,
      double square_ratio_thresh = 1.5)
  {
    if (pc_front.points_.size() < 10)
    {
      // Not enough points → treat as square (safe default)
      return true;
    }

    // Accumulate covariance in plane
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();

    for (const auto &p : pc_front.points_)
    {
      Eigen::Vector3d d = p - c_front;

      // project into plane
      Eigen::Vector3d d_plane =
          d - d.dot(n_front) * n_front;

      C += d_plane * d_plane.transpose();
    }

    C /= static_cast<double>(pc_front.points_.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
    if (es.info() != Eigen::Success)
    {
      return true;
    }

    Eigen::Vector3d evals = es.eigenvalues();

    // largest / second largest
    double lambda1 = evals(2);
    double lambda2 = evals(1);

    if (lambda2 < 1e-9)
    {
      return true;
    }

    double ratio = lambda1 / lambda2;

    return ratio < square_ratio_thresh;
  }

  inline Eigen::Vector3d closest_point_between_lines(
      const Eigen::Vector3d &c1,
      const Eigen::Vector3d &n1,
      const Eigen::Vector3d &c2,
      const Eigen::Vector3d &n2)
  {
    Eigen::Vector3d r = c1 - c2;

    double a = n1.dot(n1);
    double b = n1.dot(n2);
    double c = n2.dot(n2);
    double d = n1.dot(r);
    double e = n2.dot(r);

    double denom = a * c - b * b;
    if (std::abs(denom) < 1e-6)
    {
      // Lines almost parallel → fallback to average
      return 0.5 * (c1 + c2);
    }

    double t = (b * e - c * d) / denom;
    double s = (a * e - b * d) / denom;

    Eigen::Vector3d p1 = c1 + t * n1;
    Eigen::Vector3d p2 = c2 + s * n2;

    return 0.5 * (p1 + p2);
  }

  inline Eigen::Matrix3d build_frame_from_planes(
      const Eigen::Vector3d &n_top,
      const Eigen::Vector3d &n_front,
      bool front_is_square)
  {
    Eigen::Vector3d z = n_top.normalized();

    Eigen::Vector3d x, y;

    if (front_is_square)
    {
      // n_front defines x-axis
      x = n_front.normalized();
      y = z.cross(x).normalized();
    }
    else
    {
      // n_front defines y-axis
      y = n_front.normalized();
      x = y.cross(z).normalized();
    }

    // Re-orthogonalize
    y = z.cross(x).normalized();

    Eigen::Matrix3d R;
    R.col(0) = x;
    R.col(1) = y;
    R.col(2) = z;
    return R;
  }

  struct PlaneSelection
  {
    Eigen::Vector3d n_top, c_top;
    Eigen::Vector3d n_front, c_front;
    bool success = false;
  };

  PlaneSelection select_top_and_front_planes(
      const std::vector<std::pair<Eigen::Vector4d,
                                  geometry::PointCloud>> &planes,
      const Eigen::Vector3d &z_world,
      double angle_thresh,
      double max_plane_center_dist)
  {
    PlaneSelection sel;

    bool found_top = false;

    struct Candidate
    {
      Eigen::Vector3d n_raw;
      Eigen::Vector3d c;
      size_t support;
    };

    std::vector<Candidate> fronts;

    // ----------------------------------------------------------
    // Classify planes
    // ----------------------------------------------------------
    for (const auto &[plane, pc] : planes)
    {

      Eigen::Vector3d n_raw = plane.head<3>().normalized();
      Eigen::Vector3d c = compute_center(pc);

      double cos_z = std::abs(n_raw.dot(z_world));

      // -----------------------
      // Top plane
      // -----------------------
      if (!found_top && cos_z > angle_thresh)
      {
        sel.n_top = orient_normal_towards(plane, z_world);
        sel.c_top = c;
        found_top = true;
        continue;
      }

      // -----------------------
      // Front plane candidates
      // -----------------------
      if (cos_z < 0.2)
      {
        fronts.push_back({n_raw, c, pc.points_.size()});
      }
    }

    if (!found_top || fronts.empty())
    {
      return sel;
    }

    // ----------------------------------------------------------
    // Sort front candidates by support
    // ----------------------------------------------------------
    std::sort(
        fronts.begin(), fronts.end(),
        [](const auto &a, const auto &b)
        {
          return a.support > b.support;
        });

    // ----------------------------------------------------------
    // Distance gating + orientation
    // ----------------------------------------------------------
    for (const auto &fc : fronts)
    {

      if ((fc.c - sel.c_top).norm() < max_plane_center_dist)
      {

        // Vector from plane center towards sensor (origin)
        Eigen::Vector3d to_sensor = -fc.c.normalized();

        sel.n_front =
            (fc.n_raw.dot(to_sensor) < 0.0)
                ? -fc.n_raw
                : fc.n_raw;

        sel.c_front = fc.c;
        sel.success = true;
        return sel;
      }
    }

    return sel;
  }

  // ============================================================
  // GLOBAL REGISTRATION
  // ============================================================

  GlobalRegistrationResult compute_global_registration(
      const geometry::PointCloud &scene,
      const Eigen::Vector3d &z_world,
      double angle_thresh,
      int max_planes,
      double dist_thresh,
      int min_inliers,
      double max_plane_center_dist)
  {
    GlobalRegistrationResult out;

    // ----------------------------------------------------------
    // Scene centroid (debug / sanity)
    // ----------------------------------------------------------
    out.center = compute_center(scene);

    // ----------------------------------------------------------
    // Plane extraction
    // ----------------------------------------------------------
    out.planes =
        extract_planes(
            scene, max_planes,
            dist_thresh, min_inliers);

    if (out.planes.empty())
    {
      std::cerr << "[globreg] no planes detected\n";
      return out;
    }

    out.num_planes = static_cast<int>(out.planes.size());

    // ----------------------------------------------------------
    // Select planes
    // ----------------------------------------------------------
    auto sel = select_top_and_front_planes(
        out.planes, z_world,
        angle_thresh, max_plane_center_dist);

    if (!sel.success)
    {
      std::cerr << "[globreg] plane selection failed\n";
      return out;
    }

    out.n_top = sel.n_top;
    out.c_top = sel.c_top;
    out.front_normals = {sel.n_front};
    out.front_centers = {sel.c_front};

    // ----------------------------------------------------------
    // Determine front plane shape (PCA)
    // ----------------------------------------------------------
    const geometry::PointCloud *pc_front = nullptr;

    for (const auto &[plane, pc] : out.planes)
    {
      Eigen::Vector3d c = compute_center(pc);
      if ((c - sel.c_front).norm() < 1e-6)
      {
        pc_front = &pc;
        break;
      }
    }

    bool front_is_square = true;
    if (pc_front)
    {
      front_is_square =
          is_front_plane_square_pca(
              *pc_front,
              sel.c_front,
              sel.n_front);
    }

    // ----------------------------------------------------------
    // Orientation
    // ----------------------------------------------------------
    out.R_base = build_frame_from_planes(
      sel.n_top, sel.n_front, front_is_square);

    // ----------------------------------------------------------
    // Translation via LS line intersection
    // ----------------------------------------------------------
    Eigen::Vector3d center =
        closest_point_between_lines(
            sel.c_top, sel.n_top,
            sel.c_front, sel.n_front);

    out.center = center;

    // ----------------------------------------------------------
    // Concatenate plane inliers (outlier-free cloud)
    // ----------------------------------------------------------
    out.plane_cloud =
        std::make_shared<geometry::PointCloud>();

    for (const auto &[plane, pc] : out.planes)
    {
      *out.plane_cloud += pc;
    }

    if (out.plane_cloud->points_.empty())
    {
      std::cerr << "[globreg] plane_cloud empty\n";
      return out;
    }

    out.success = true;
    return out;
  }

  // ============================================================
  // LOCAL REGISTRATION Helper Functions
  // ============================================================

  double geometry_score(
      const TemplateData &tpl,
      const Eigen::Vector3d &n_top,
      const Eigen::Vector3d &n_front,
      int face_index)
  {
    const Eigen::Vector3d &tpl_side_n =
        (face_index == 0) ? tpl.normal_long : tpl.normal_short;

    double top_align =
        std::abs(tpl.normal_top.dot(n_top));

    double side_align =
        std::abs(tpl_side_n.dot(n_front));

    // Hard reject impossible hypotheses
    if (top_align < 0.95 || side_align < 0.85)
    {
      return -1.0;
    }

    // Soft score
    return top_align + side_align;
  }

  Eigen::Matrix4d rotate_transform_yaw(
      const Eigen::Matrix4d &T,
      double yaw_rad)
  {
    Eigen::Matrix4d Rz = Eigen::Matrix4d::Identity();
    Rz.block<3, 3>(0, 0) =
        Eigen::AngleAxisd(
            yaw_rad, Eigen::Vector3d::UnitZ())
            .toRotationMatrix();

    return T * Rz; // post-multiply: rotate in local frame
  }

  pipelines::registration::RegistrationResult
  run_icp(
      const geometry::PointCloud &scene,
      const TemplateData &tpl,
      const Eigen::Matrix4d &T_init,
      double icp_dist)
  {
    return pipelines::registration::RegistrationICP(
        *tpl.pcd,
        scene,
        icp_dist,
        T_init,
        pipelines::registration::
            TransformationEstimationPointToPlane());
  }

  // ============================================================
  // LOCAL REGISTRATION
  // ============================================================
  LocalRegistrationResult compute_local_registration(
      const geometry::PointCloud &scene,
      const std::vector<TemplateData> &templates,
      const GlobalRegistrationResult &glob,
      double icp_dist)
  {
    LocalRegistrationResult best;
    best.icp.fitness_ = -1.0;

    if (!glob.success)
    {
      return best;
    }

    // constexpr double W_GEOM = 1.0;
    constexpr double W_ICP = 0.5;

    for (size_t ti = 0; ti < templates.size(); ++ti)
    {
      const auto &tpl = templates[ti];

      if (tpl.num_faces != glob.num_planes)
      {
        // std::cout << "tpl.num_faces " << tpl.num_faces << " != glob.num_faces " << glob.num_planes << std::endl;
        continue;
      }
      // else
      // {
      //   std::cout << "template [" << ti << "] with " << tpl.num_faces << " faces" << std::endl;
      // }

      Eigen::Matrix4d T_base =
          globalResultToTransform(glob);

      // test against yaw hypothesis
      for (int yaw_deg = 0; yaw_deg < 360; yaw_deg += 90)
      {
        // --------------------------------------------------
        // Build transform hypothesis
        // --------------------------------------------------
        double yaw_rad = yaw_deg * DEG2RAD;

        // --------------------------------------------------
        // Build yaw hypothesis
        // --------------------------------------------------
        Eigen::Matrix4d T_init =
            rotate_transform_yaw(T_base, yaw_rad);

        // --------------------------------------------------
        // ICP refinement
        // --------------------------------------------------
        auto icp =
            run_icp(scene, tpl, T_init, icp_dist);

        if (icp.fitness_ <= 0.0)
        {
          continue;
        }

        // --------------------------------------------------
        // Combined score
        // --------------------------------------------------
        double score =
            W_ICP * icp.fitness_;

        if (!best.success || score > best.score)
        {
          best.success = true;
          best.score = score;
          best.template_name = tpl.name;
          best.template_index = static_cast<int>(ti);
          best.icp = icp;
        }
      }
    }

    return best;
  }

  // ============================================================
  // Utility
  // ============================================================

  Eigen::Matrix4d
  globalResultToTransform(const GlobalRegistrationResult &glob)
  {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    if (!glob.success)
    {
      return T;
    }

    T.block<3, 3>(0, 0) = glob.R_base;
    T.block<3, 1>(0, 3) = glob.center;

    return T;
  }

} // namespace pcd_block
