#include "pcd_block_estimation/utils.hpp"

namespace pcd_block
{

Eigen::Vector3d normalize(const Eigen::Vector3d &v)
{
    return v.normalized();
}

Eigen::Vector3d compute_center(
    const open3d::geometry::PointCloud &pcd)
{
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (const auto &p : pcd.points_)
        center += p;
    return center / static_cast<double>(pcd.points_.size());
}

std::vector<std::pair<Eigen::Vector4d,
                      open3d::geometry::PointCloud>>
extract_planes(const open3d::geometry::PointCloud &input,
               int max_planes,
               double dist_thresh,
               int min_inliers)
{
    std::vector<std::pair<Eigen::Vector4d,
                          open3d::geometry::PointCloud>> planes;

    open3d::geometry::PointCloud rest = input;

    for (int i = 0; i < max_planes; ++i)
    {
        if (rest.points_.size() < min_inliers)
            break;

        Eigen::Vector4d plane;
        std::vector<size_t> indices;

        std::tie(plane, indices) =
            rest.SegmentPlane(dist_thresh, 3, 2000);

        if (indices.size() < min_inliers)
            break;

        auto pc = *rest.SelectByIndex(indices);
        planes.emplace_back(plane, pc);

        rest = *rest.SelectByIndex(indices, true);
    }

    return planes;
}

} // namespace pcd_block
