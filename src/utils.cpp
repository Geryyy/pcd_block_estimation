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
                              open3d::geometry::PointCloud>>
            planes;

        open3d::geometry::PointCloud rest = input;

        for (int i = 0; i < max_planes; ++i)
        {

            if (rest.points_.size() < static_cast<size_t>(min_inliers))
            {
                break;
            }

            Eigen::Vector4d plane;
            std::vector<size_t> indices;

            std::tie(plane, indices) =
                rest.SegmentPlane(dist_thresh, 3, 2000);

            if (indices.size() < static_cast<size_t>(min_inliers))
            {
                break;
            }

            // --------------------------------------------------
            // Extract raw plane inliers
            // --------------------------------------------------
            open3d::geometry::PointCloud pc =
                *rest.SelectByIndex(indices);

            // params:
            // const double support_radius,
            // const int min_neighbors,
            // const int erosion_iters
            // // --------------------------------------------------
            // // NEW: local support pruning via KD-tree
            // // --------------------------------------------------
            // for (int iter = 0; iter < erosion_iters; ++iter)
            // {

            //     if (pc.points_.size() < static_cast<size_t>(min_inliers))
            //     {
            //         break;
            //     }

            //     open3d::geometry::KDTreeFlann kdtree(pc);
            //     std::vector<Eigen::Vector3d> kept;

            //     for (size_t pi = 0; pi < pc.points_.size(); ++pi)
            //     {

            //         std::vector<int> nn_indices;
            //         std::vector<double> nn_dists;

            //         int k = kdtree.SearchRadius(
            //             pc.points_[pi],
            //             support_radius,
            //             nn_indices,
            //             nn_dists);

            //         // k includes the point itself
            //         if (k >= min_neighbors)
            //         {
            //             kept.push_back(pc.points_[pi]);
            //         }
            //     }

            //     pc.points_ = std::move(kept);
            // }

            // // --------------------------------------------------
            // // Fallback: if erosion killed too much, keep raw
            // // --------------------------------------------------
            // if (pc.points_.size() < static_cast<size_t>(min_inliers))
            // {
            //     pc = *rest.SelectByIndex(indices);
            //     std::cerr << "extract_planes(): clustering failed!" << std::endl;
            // }

            planes.emplace_back(plane, pc);

            // --------------------------------------------------
            // Remove original RANSAC inliers from rest
            // --------------------------------------------------
            rest = *rest.SelectByIndex(indices, true);
        }

        return planes;
    }

} // namespace pcd_block
