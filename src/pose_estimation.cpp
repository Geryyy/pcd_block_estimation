#include "pcd_block_estimation/pose_estimation.hpp"
#include "pcd_block_estimation/utils.hpp"

using namespace open3d;

namespace pcd_block {

PoseResult estimate_pose(
    const geometry::PointCloud &scene,
    const std::vector<TemplateData> &templates,
    const Eigen::Matrix3d &R_base,
    const Eigen::Vector3d &scene_center,
    int num_planes,
    double icp_dist,
    int yaw_step_deg)
{
    PoseResult best{};
    best.icp.fitness_ = -1.0;

    int idx = 0;
    for (const auto &tpl : templates) {
        if (tpl.num_faces != num_planes) {
            idx++; continue;
        }

        Eigen::Vector3d tpl_center =
            compute_center(*tpl.pcd);

        for (int yaw = 0; yaw < 360; yaw += yaw_step_deg) {

            Eigen::Matrix3d R_yaw =
                geometry::Geometry3D::GetRotationMatrixFromAxisAngle(
                    R_base.col(2) * yaw * M_PI / 180.0);

            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3,3>(0,0) = R_yaw * R_base;
            T.block<3,1>(0,3) =
                scene_center - T.block<3,3>(0,0) * tpl_center;

            auto result =
                pipelines::registration::RegistrationICP(
                    *tpl.pcd, scene,
                    icp_dist, T,
                    pipelines::registration::
                        TransformationEstimationPointToPlane());

            if (result.fitness_ > best.icp.fitness_) {
                best = {tpl.name, idx, yaw, result};
            }
        }
        idx++;
    }
    return best;
}

}
