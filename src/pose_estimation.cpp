#include "pcd_block_estimation/pose_estimation.hpp"
#include "pcd_block_estimation/utils.hpp"

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>

using namespace open3d;

#define GLOBREG_DBG(msg) \
  std::cerr << "[globreg][dbg] " << msg << std::endl
#define LOCREG_DBG(msg) \
  std::cerr << "[locreg][dbg] " << msg << std::endl

namespace pcd_block
{

// ============================================================
// Helpers
// ============================================================

static Eigen::Vector3d
orient_normal_towards(
  const Eigen::Vector4d & plane,
  const Eigen::Vector3d & reference_dir)
{
  Eigen::Vector3d n = plane.head<3>().normalized();

  if (n.dot(reference_dir) < 0.0) {
    n = -n;
  }
  return n;
}


static FrontPlaneShape
classify_front_plane_bb(
  const geometry::PointCloud & pc_front,
  const Eigen::Vector3d & c_front,
  const Eigen::Vector3d & n_front,
  double square_ratio_thresh = 1.5)
{
  if (pc_front.points_.size() < 20) {
    return FrontPlaneShape::SQUARE;
  }

  // --------------------------------------------------------
  // 1️⃣ Build orthonormal basis in plane
  // --------------------------------------------------------

  Eigen::Vector3d n = n_front.normalized();

  // Choose arbitrary vector not parallel to n
  Eigen::Vector3d tmp =
    std::abs(n.z()) < 0.9 ?
    Eigen::Vector3d::UnitZ() :
    Eigen::Vector3d::UnitX();

  Eigen::Vector3d u = (tmp - tmp.dot(n) * n).normalized();
  Eigen::Vector3d v = n.cross(u).normalized();

  // --------------------------------------------------------
  // 2️⃣ Project points to 2D plane coordinates
  // --------------------------------------------------------

  std::vector<double> xs;
  std::vector<double> ys;

  xs.reserve(pc_front.points_.size());
  ys.reserve(pc_front.points_.size());

  for (const auto & p : pc_front.points_) {
    Eigen::Vector3d d = p - c_front;

    xs.push_back(d.dot(u));
    ys.push_back(d.dot(v));
  }

  if (xs.size() < 10) {
    return FrontPlaneShape::SQUARE;
  }

  // --------------------------------------------------------
  // 3️⃣ Robust extents (5%–95% percentiles)
  // --------------------------------------------------------

  auto percentile = [](std::vector<double> & data, double q)
    {
      std::sort(data.begin(), data.end());
      size_t idx =
        static_cast<size_t>(q * (data.size() - 1));
      return data[idx];
    };

  std::vector<double> xs_copy = xs;
  std::vector<double> ys_copy = ys;

  double x_min = percentile(xs_copy, 0.05);
  double x_max = percentile(xs_copy, 0.95);
  double y_min = percentile(ys_copy, 0.05);
  double y_max = percentile(ys_copy, 0.95);

  double width = x_max - x_min;
  double height = y_max - y_min;

  if (width < 1e-6 || height < 1e-6) {
    return FrontPlaneShape::SQUARE;
  }

  double ratio =
    std::max(width, height) /
    std::min(width, height);

  GLOBREG_DBG("BB ratio: " << ratio);

  if (ratio < square_ratio_thresh) {
    return FrontPlaneShape::SQUARE;
  }

  // --------------------------------------------------------
  // 4️⃣ Determine elongation direction
  // --------------------------------------------------------

  const Eigen::Vector3d Z_WORLD(0.0, -1.0, 0.0);

  // If vertical axis in plane aligns with Z_WORLD
  double vertical_alignment =
    std::abs(v.dot(Z_WORLD));

  bool height_is_vertical =
    vertical_alignment > 0.7;

  GLOBREG_DBG("Vertical alignment: " << vertical_alignment);

  if (height_is_vertical) {
    if (height > width) {
      std::cerr << "Case 1: Classified as tall vertical" << std::endl;
      return FrontPlaneShape::TALL_VERTICAL;
    } else {
      return FrontPlaneShape::WIDE_HORIZONTAL;
    }
  } else {
    if (width > height) {
      return FrontPlaneShape::WIDE_HORIZONTAL;
    } else {
      std::cerr << "Case 2: Classified as tall vertical" << std::endl;
      return FrontPlaneShape::TALL_VERTICAL;
    }
  }
}

// TODO: square_ratio_thresh as parameter
static FrontPlaneShape
classify_front_plane_pca(
  const geometry::PointCloud & pc_front,
  const Eigen::Vector3d & c_front,
  const Eigen::Vector3d & n_front,
  double square_ratio_thresh = 1.5)
{

  if (pc_front.points_.size() < 10) {
    return FrontPlaneShape::SQUARE;
  }

  // --------------------------------------------------------
  // Covariance in plane
  // --------------------------------------------------------
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();

  for (const auto & p : pc_front.points_) {
    Eigen::Vector3d d = p - c_front;

    // project into plane
    Eigen::Vector3d d_plane =
      d - d.dot(n_front) * n_front;

    C += d_plane * d_plane.transpose();
  }

  C /= static_cast<double>(pc_front.points_.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
  if (es.info() != Eigen::Success) {
    return FrontPlaneShape::SQUARE;
  }

  // Eigenvalues sorted ascending
  Eigen::Vector3d evals = es.eigenvalues();
  Eigen::Matrix3d evecs = es.eigenvectors();

  double lambda1 = evals(2);   // largest
  double lambda2 = evals(1);   // second largest

  if (lambda2 < 1e-9) {
    return FrontPlaneShape::SQUARE;
  }

  double ratio = lambda1 / lambda2;

  GLOBREG_DBG("PCA ratio: " << ratio);

  if (ratio < square_ratio_thresh) {
    return FrontPlaneShape::SQUARE;
  }

  // --------------------------------------------------------
  // Determine elongation direction
  // --------------------------------------------------------

  // principal direction of largest variance
  Eigen::Vector3d major_axis = evecs.col(2).normalized();

  // remove potential sign ambiguity
  if (major_axis.dot(n_front) > 0.99) {
    // shouldn't happen but safe guard
    return FrontPlaneShape::SQUARE;
  }

  // compare against vertical direction
  const Eigen::Vector3d Z_WORLD(0.0, -1.0, 0.0);

  double vertical_alignment =
    std::abs(major_axis.dot(Z_WORLD));

  if (vertical_alignment > 0.7) {
    // elongated in vertical direction
    return FrontPlaneShape::TALL_VERTICAL;
  } else {
    // elongated horizontally
    return FrontPlaneShape::WIDE_HORIZONTAL;
  }
}

inline Eigen::Vector3d closest_point_between_lines(
  const Eigen::Vector3d & c1,
  const Eigen::Vector3d & n1,
  const Eigen::Vector3d & c2,
  const Eigen::Vector3d & n2)
{
  Eigen::Vector3d r = c1 - c2;

  double a = n1.dot(n1);
  double b = n1.dot(n2);
  double c = n2.dot(n2);
  double d = n1.dot(r);
  double e = n2.dot(r);

  double denom = a * c - b * b;
  if (std::abs(denom) < 1e-6) {
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
  const Eigen::Vector3d & n_top,
  const Eigen::Vector3d & n_front,
  bool front_is_square)
{
  Eigen::Vector3d z = n_top.normalized();

  Eigen::Vector3d x, y;

  if (front_is_square) {
    // n_front defines x-axis
    x = n_front.normalized();
    y = z.cross(x).normalized();
  } else {
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

static std::shared_ptr<geometry::PointCloud>
clip_scene_with_top_and_front(
  const geometry::PointCloud & scene,
  Eigen::Vector3d c_front,
  Eigen::Vector3d n_front,
  const Eigen::Vector3d & c_top,
  const Eigen::Vector3d & n_top,
  double front_margin,
  double top_margin)
{
  auto clipped =
    std::make_shared<geometry::PointCloud>();

  clipped->points_.reserve(scene.points_.size());

  // ----------------------------------------------------------
  // Ensure front normal points toward sensor origin
  // ----------------------------------------------------------

  Eigen::Vector3d origin = Eigen::Vector3d::Zero();

  if ((origin - c_front).dot(n_front) < 0.0) {
    n_front = -n_front;
  }

  // ----------------------------------------------------------
  // Half-space clipping
  // Keep:
  //   - points behind front plane
  //   - points below top plane
  // ----------------------------------------------------------

  for (const auto & p : scene.points_) {
    double dist_front =
      (p - c_front).dot(n_front);

    double dist_top =
      (p - c_top).dot(n_top);

    bool keep =
      (dist_front <= front_margin) &&
      (dist_top <= top_margin);

    if (keep) {
      clipped->points_.push_back(p);
    }
  }

  return clipped;
}

struct PlaneSelection
{
  Eigen::Vector3d n_top, c_top;
  Eigen::Vector3d n_front, c_front;
  bool success = false;
};

PlaneSelection select_top_and_front_planes(
  const std::vector<std::pair<Eigen::Vector4d,
  geometry::PointCloud>> & planes,
  const Eigen::Vector3d & z_world,
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
  for (const auto &[plane, pc] : planes) {

    Eigen::Vector3d n_raw = plane.head<3>().normalized();
    Eigen::Vector3d c = compute_center(pc);

    double cos_z = std::abs(n_raw.dot(z_world));

    // -----------------------
    // Top plane
    // -----------------------
    if (!found_top && cos_z > angle_thresh) {
      sel.n_top = orient_normal_towards(plane, z_world);
      sel.c_top = c;
      found_top = true;
      continue;
    }

    // -----------------------
    // Front plane candidates
    // -----------------------
    if (cos_z < 0.2) {
      fronts.push_back({n_raw, c, pc.points_.size()});
    }
  }

  if (!found_top) {
    GLOBREG_DBG("No top plane candidate found");
    return sel;
  }

  if (fronts.empty()) {
    GLOBREG_DBG("No front plane candidates found");
    return sel;
  }

  // ----------------------------------------------------------
  // Sort front candidates by support
  // ----------------------------------------------------------
  std::sort(
    fronts.begin(), fronts.end(),
    [](const auto & a, const auto & b)
    {
      return a.support > b.support;
    });

  // ----------------------------------------------------------
  // Distance gating + orientation
  // ----------------------------------------------------------
  for (const auto & fc : fronts) {

    if ((fc.c - sel.c_top).norm() < max_plane_center_dist) {

      // Vector from plane center towards sensor (origin)
      Eigen::Vector3d to_sensor = -fc.c.normalized();

      sel.n_front =
        (fc.n_raw.dot(to_sensor) < 0.0) ?
        -fc.n_raw :
        fc.n_raw;

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
  const geometry::PointCloud & scene,
  const Eigen::Vector3d & z_world,
  double angle_thresh,
  int max_planes,
  double dist_thresh,
  int min_inliers,
  double max_plane_center_dist,
  bool enable_plane_clipping,
  bool reject_tall_vertical)
{
  GlobalRegistrationResult out_initial;
  bool clipping_enabled = enable_plane_clipping;


  GlobalRegistrationResult out;
  geometry::PointCloud scene_filtered;

  if (clipping_enabled) {
    // ----------------------------------------------------------
    // Scene centroid
    // ----------------------------------------------------------
    out_initial.center = compute_center(scene);

    // ----------------------------------------------------------
    // Plane extraction
    // ----------------------------------------------------------
    out_initial.planes =
      extract_planes(
      scene, max_planes,
      dist_thresh, min_inliers);

    if (out_initial.planes.empty()) {
      GLOBREG_DBG("FAIL: no_planes");
      return out_initial;
    }

    out_initial.num_planes = static_cast<int>(out_initial.planes.size());

    // ----------------------------------------------------------
    // Select top + front planes
    // ----------------------------------------------------------
    auto sel_initial = select_top_and_front_planes(
      out_initial.planes, z_world,
      angle_thresh, max_plane_center_dist);

    if (!sel_initial.success) {
      GLOBREG_DBG("FAIL: initial select_planes");
      return out_initial;
    }

    out_initial.n_top = sel_initial.n_top;
    out_initial.c_top = sel_initial.c_top;
    out_initial.front_normals = {sel_initial.n_front};
    out_initial.front_centers = {sel_initial.c_front};

    // -----------------------------------------------------------
    // Clipping scene to planes can improve ICP stability by removing outliers and reducing noise, but risks removing too much data if planes are small or far from each other. Make this optional for now.
    // -----------------------------------------------------------
    constexpr double FRONT_CLIP_MARGIN = 0.05; // keep points up to 5cm in front of the front plane
    constexpr double TOP_CLIP_MARGIN = 0.05;   // keep points up to

    auto clipped =
      clip_scene_with_top_and_front(
      scene,
      sel_initial.c_front,
      sel_initial.n_front,
      sel_initial.c_top,
      sel_initial.n_top,
      FRONT_CLIP_MARGIN,
      TOP_CLIP_MARGIN);

    if (clipped->points_.size() > 50) {
      GLOBREG_DBG(
        "Clipping removed "
          << scene.points_.size() - clipped->points_.size()
          << " points");

    } else {
      GLOBREG_DBG("Clipping skipped (too few points left)");
    }

    scene_filtered = *clipped;
  } else {
    scene_filtered = scene;
  }


  // ----------------------------------------------------------
  // Scene centroid
  // ----------------------------------------------------------
  out.center = compute_center(scene_filtered);

  // ----------------------------------------------------------
  // Plane extraction
  // ----------------------------------------------------------
  out.planes =
    extract_planes(
    scene_filtered, max_planes,
    dist_thresh, min_inliers);

  if (out.planes.empty()) {
    GLOBREG_DBG("FAIL: no_planes");
    return out;
  }

  out.num_planes = static_cast<int>(out.planes.size());

  // ----------------------------------------------------------
  // Select top + front planes
  // ----------------------------------------------------------
  auto sel = select_top_and_front_planes(
    out.planes, z_world,
    angle_thresh, max_plane_center_dist);

  if (!sel.success) {
    GLOBREG_DBG("FAIL: select_planes");
    return out;
  }

  out.n_top = sel.n_top;
  out.c_top = sel.c_top;
  out.front_normals = {sel.n_front};
  out.front_centers = {sel.c_front};


  const geometry::PointCloud * pc_front = nullptr;

  for (const auto &[plane, pc] : out.planes) {
    if ((compute_center(pc) - sel.c_front).norm() < 1e-6) {
      pc_front = &pc;
      break;
    }
  }

  if (!pc_front) {
    GLOBREG_DBG("FAIL: no front plane cloud found");
    return out;
  }

  FrontPlaneShape shape =
    classify_front_plane_pca(*pc_front, sel.c_front, sel.n_front);

  bool front_is_square;

  if (shape == FrontPlaneShape::SQUARE) {
    GLOBREG_DBG("Front plane classified as SQUARE");
    front_is_square = true;
  } else if (shape == FrontPlaneShape::WIDE_HORIZONTAL) {
    // side of block
    GLOBREG_DBG("Front plane classified as WIDE_HORIZONTAL");
    front_is_square = false;
  } else if (shape == FrontPlaneShape::TALL_VERTICAL) {
    // e.g. stacked blocks or partial-occlusion artifacts in grasped mode
    if (reject_tall_vertical) {
      GLOBREG_DBG("FAIL: Front plane classified as TALL_VERTICAL");
      return out;
    }
    GLOBREG_DBG("Front plane classified as TALL_VERTICAL (accepted by config)");
    front_is_square = false;
  }

  // ----------------------------------------------------------
  // Orientation
  // ----------------------------------------------------------
  out.R_base =
    build_frame_from_planes(
    sel.n_top, sel.n_front, front_is_square);

  // ----------------------------------------------------------
  // Translation
  // ----------------------------------------------------------
  out.center =
    closest_point_between_lines(
    sel.c_top, sel.n_top,
    sel.c_front, sel.n_front);

  // ----------------------------------------------------------
  // Plane cloud
  // ----------------------------------------------------------
  out.plane_cloud =
    std::make_shared<geometry::PointCloud>();

  for (const auto &[plane, pc] : out.planes) {
    *out.plane_cloud += pc;
  }

  if (out.plane_cloud->points_.empty()) {
    GLOBREG_DBG("FAIL: empty_plane_cloud");
    return out;
  }

  out.success = true;

  // ----------------------------------------------------------
  // Success summary
  // ----------------------------------------------------------
  GLOBREG_DBG(
    "OK: planes=" << out.num_planes
                  << " front=" << (front_is_square ? "square" : "rect"));

  return out;
}

// ============================================================
// LOCAL REGISTRATION Helper Functions
// ============================================================

double geometry_score(
  const TemplateData & tpl,
  const Eigen::Vector3d & n_top,
  const Eigen::Vector3d & n_front,
  int face_index)
{
  const Eigen::Vector3d & tpl_side_n =
    (face_index == 0) ? tpl.normal_long : tpl.normal_short;

  double top_align =
    std::abs(tpl.normal_top.dot(n_top));

  double side_align =
    std::abs(tpl_side_n.dot(n_front));

  // Hard reject impossible hypotheses
  if (top_align < 0.95 || side_align < 0.85) {
    return -1.0;
  }

  // Soft score
  return top_align + side_align;
}

Eigen::Matrix4d rotate_transform_yaw(
  const Eigen::Matrix4d & T,
  double yaw_rad)
{
  Eigen::Matrix4d Rz = Eigen::Matrix4d::Identity();
  Rz.block<3, 3>(0, 0) =
    Eigen::AngleAxisd(
    yaw_rad, Eigen::Vector3d::UnitZ())
    .toRotationMatrix();

  return T * Rz;   // post-multiply: rotate in local frame
}

pipelines::registration::RegistrationResult
run_icp(
  const geometry::PointCloud & scene,
  const TemplateData & tpl,
  const Eigen::Matrix4d & T_init,
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

pipelines::registration::RegistrationResult
run_icp_point_to_point(
  const geometry::PointCloud & scene,
  const TemplateData & tpl,
  const Eigen::Matrix4d & T_init,
  double icp_dist)
{
  return pipelines::registration::RegistrationICP(
    *tpl.pcd,
    scene,
    icp_dist,
    T_init,
    pipelines::registration::
    TransformationEstimationPointToPoint());
}

// ============================================================
// LOCAL REGISTRATION
// ============================================================
LocalRegistrationResult compute_local_registration(
  const geometry::PointCloud & scene,
  const std::vector<TemplateData> & templates,
  const GlobalRegistrationResult & glob,
  double icp_dist,
  bool relax_num_faces_match,
  const Eigen::Vector3d * translation_seed_world,
  const std::vector<double> & icp_dist_multipliers,
  bool enable_point_to_point_fallback)
{
  LocalRegistrationResult best;
  best.icp.fitness_ = -1.0;
  best.templates_total = templates.size();

  if (!glob.success) {
    best.failure_reason = "global registration not successful";
    LOCREG_DBG("LOCAL FAIL: global registration not successful");
    return best;
  }

  LOCREG_DBG(
    "LOCAL start: scene_points=" << scene.points_.size() <<
      " templates=" << templates.size() <<
      " glob.num_planes=" << glob.num_planes <<
      " relax_num_faces_match=" << (relax_num_faces_match ? "true" : "false") <<
      " use_translation_seed=" << (translation_seed_world ? "true" : "false") <<
      " p2p_fallback=" << (enable_point_to_point_fallback ? "true" : "false") <<
      " dist_multipliers=" << icp_dist_multipliers.size() <<
      " icp_dist=" << icp_dist);

  open3d::geometry::PointCloud scene_with_normals = scene;

  if (!scene_with_normals.HasNormals()) {
    scene_with_normals.EstimateNormals(
      open3d::geometry::KDTreeSearchParamHybrid(0.05, 30));
  }

  if (!scene_with_normals.HasNormals()) {
    scene_with_normals.EstimateNormals(
      open3d::geometry::KDTreeSearchParamHybrid(
        /* radius = */ 0.05,
        /* max_nn = */ 30));
  }

  // constexpr double W_GEOM = 1.0;
  constexpr double W_ICP = 0.5;

  size_t templates_tested = 0;
  size_t templates_skipped_num_faces = 0;
  size_t icp_attempts = 0;
  size_t icp_positive = 0;
  double best_fitness_seen = -std::numeric_limits<double>::infinity();
  double best_rmse_seen = std::numeric_limits<double>::infinity();

  for (size_t ti = 0; ti < templates.size(); ++ti) {
    const auto & tpl = templates[ti];

    if (!relax_num_faces_match && tpl.num_faces != glob.num_planes) {
      templates_skipped_num_faces++;
      continue;
    }
    templates_tested++;

    Eigen::Matrix4d T_base =
      globalResultToTransform(glob);
    if (translation_seed_world) {
      T_base.block<3, 1>(0, 3) = *translation_seed_world;
    }

    // test against yaw hypothesis
    for (int yaw_deg = 0; yaw_deg < 360; yaw_deg += 360) {
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
      pipelines::registration::RegistrationResult icp;
      icp.fitness_ = -1.0;
      for (double dist_scale : icp_dist_multipliers) {
        const double dist = std::max(1e-4, icp_dist * dist_scale);
        auto icp_p2l = run_icp(scene_with_normals, tpl, T_init, dist);
        icp_attempts++;
        if (icp_p2l.fitness_ > icp.fitness_) {
          icp = icp_p2l;
        }
        if (std::isfinite(icp_p2l.fitness_)) {
          best_fitness_seen = std::max(best_fitness_seen, icp_p2l.fitness_);
        }
        if (std::isfinite(icp_p2l.inlier_rmse_)) {
          best_rmse_seen = std::min(best_rmse_seen, icp_p2l.inlier_rmse_);
        }
        if (icp_p2l.fitness_ > 0.0) {
          LOCREG_DBG(
            "LOCAL candidate: tpl=" << tpl.name <<
              " yaw=" << yaw_deg <<
              " dist=" << dist <<
              " method=point_to_plane" <<
              " fitness=" << icp_p2l.fitness_ <<
              " rmse=" << icp_p2l.inlier_rmse_);
          icp_positive++;
          break;
        }

        if (!enable_point_to_point_fallback) {
          continue;
        }

        auto icp_p2p = run_icp_point_to_point(scene_with_normals, tpl, T_init, dist);
        icp_attempts++;
        if (icp_p2p.fitness_ > icp.fitness_) {
          icp = icp_p2p;
        }
        if (std::isfinite(icp_p2p.fitness_)) {
          best_fitness_seen = std::max(best_fitness_seen, icp_p2p.fitness_);
        }
        if (std::isfinite(icp_p2p.inlier_rmse_)) {
          best_rmse_seen = std::min(best_rmse_seen, icp_p2p.inlier_rmse_);
        }
        if (icp_p2p.fitness_ <= 0.0) {
          continue;
        }

        auto icp_refined = run_icp(scene_with_normals, tpl, icp_p2p.transformation_, dist);
        icp_attempts++;
        if (icp_refined.fitness_ > icp.fitness_) {
          icp = icp_refined;
        }
        if (std::isfinite(icp_refined.fitness_)) {
          best_fitness_seen = std::max(best_fitness_seen, icp_refined.fitness_);
        }
        if (std::isfinite(icp_refined.inlier_rmse_)) {
          best_rmse_seen = std::min(best_rmse_seen, icp_refined.inlier_rmse_);
        }
        if (icp_refined.fitness_ > 0.0) {
          LOCREG_DBG(
            "LOCAL candidate: tpl=" << tpl.name <<
              " yaw=" << yaw_deg <<
              " dist=" << dist <<
              " method=p2p->p2l" <<
              " fitness=" << icp_refined.fitness_ <<
              " rmse=" << icp_refined.inlier_rmse_);
          icp_positive++;
          break;
        }
      }

      if (icp.fitness_ <= 0.0) {
        LOCREG_DBG(
          "LOCAL reject: tpl=" << tpl.name <<
            " yaw=" << yaw_deg <<
            " fitness=" << icp.fitness_ <<
            " rmse=" << icp.inlier_rmse_);
        continue;
      }

      // --------------------------------------------------
      // Combined score
      // --------------------------------------------------
      double score =
        W_ICP * icp.fitness_;

      if (!best.success || score > best.score) {
        best.success = true;
        best.score = score;
        best.template_name = tpl.name;
        best.template_index = static_cast<int>(ti);
        best.icp = icp;
        LOCREG_DBG(
          "LOCAL best update: tpl=" << best.template_name <<
            " idx=" << best.template_index <<
            " score=" << best.score <<
            " fitness=" << best.icp.fitness_ <<
            " rmse=" << best.icp.inlier_rmse_);
      }
    }
  }

  best.templates_tested = templates_tested;
  best.templates_skipped_num_faces = templates_skipped_num_faces;
  best.icp_attempts = icp_attempts;
  best.icp_positive = icp_positive;
  best.best_fitness_seen = best_fitness_seen;
  best.best_rmse_seen = best_rmse_seen;

  if (!best.success) {
    if (templates.empty()) {
      best.failure_reason = "no templates available";
    } else if (templates_tested == 0) {
      best.failure_reason =
        "no template matched num_faces (glob.num_planes=" + std::to_string(glob.num_planes) + ")";
    } else if (icp_attempts == 0) {
      best.failure_reason = "no ICP attempts executed";
    } else if (icp_positive == 0) {
      best.failure_reason = "all ICP attempts returned fitness <= 0";
    } else {
      best.failure_reason = "no local registration hypothesis selected";
    }
    LOCREG_DBG(
      "LOCAL FAIL summary: templates_tested=" << templates_tested <<
        " templates_skipped_num_faces=" << templates_skipped_num_faces <<
        " icp_attempts=" << icp_attempts <<
        " icp_positive=" << icp_positive <<
        " best_fitness_seen=" << best_fitness_seen <<
        " best_rmse_seen=" << best_rmse_seen <<
        " reason=" << best.failure_reason);
  } else {
    best.failure_reason.clear();
    LOCREG_DBG(
      "LOCAL OK summary: templates_tested=" << templates_tested <<
        " templates_skipped_num_faces=" << templates_skipped_num_faces <<
        " icp_attempts=" << icp_attempts <<
        " icp_positive=" << icp_positive <<
        " winner=" << best.template_name <<
        " fitness=" << best.icp.fitness_ <<
        " rmse=" << best.icp.inlier_rmse_);
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

  if (!glob.success) {
    return T;
  }

  T.block<3, 3>(0, 0) = glob.R_base;
  T.block<3, 1>(0, 3) = glob.center;

  return T;
}

} // namespace pcd_block
