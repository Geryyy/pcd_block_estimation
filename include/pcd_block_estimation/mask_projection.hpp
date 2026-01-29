#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

namespace pcd_block {

    /**
 * @brief Select point cloud points whose image projections fall inside a mask.
 *
 * Frames:
 *  - Input points: point cloud frame P
 *  - Projection: camera frame C
 *  - Output points: point cloud frame P
 *
 * @param pts   3D points in point cloud frame P
 * @param mask  Binary image mask (H×W), non-zero = valid
 * @param K     Camera intrinsics (3×3)
 * @param T_P_C Extrinsic transform from point cloud frame P → camera frame C
 * @param z_min Minimum depth in camera frame
 *
 * @return Selected points in original point cloud frame P
 */
std::vector<Eigen::Vector3d>
select_points_by_mask(
    const std::vector<Eigen::Vector3d> &pts,
    const cv::Mat &mask,
    const Eigen::Matrix3d &K,
    const Eigen::Matrix4d &T_P_C,
    double z_min = 0.1);

/**
 * @brief Compute the centroid of a binary mask in image coordinates.
 *
 * @return (u, v) pixel coordinates
 */
Eigen::Vector2i
mask_center_uv(const cv::Mat &mask);

} // namespace pcd_block
