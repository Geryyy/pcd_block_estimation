#pragma once

#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <vector>

namespace pcd_block
{

    Eigen::Vector3d normalize(const Eigen::Vector3d &v);

    Eigen::Vector3d compute_center(
        const open3d::geometry::PointCloud &pcd);

    std::vector<std::pair<Eigen::Vector4d,
                          open3d::geometry::PointCloud>>
    extract_planes(const open3d::geometry::PointCloud &input,
                   int max_planes,
                   double dist_thresh,
                   int min_inliers);

}
