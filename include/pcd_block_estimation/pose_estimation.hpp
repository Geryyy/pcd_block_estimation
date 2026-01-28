#pragma once
#include "pcd_block_estimation/template_utils.hpp"
#include <Eigen/Dense>

namespace pcd_block {

struct PoseResult {
    std::string template_name;
    int template_index;
    int yaw_deg;
    open3d::pipelines::registration::RegistrationResult icp;
};

PoseResult estimate_pose(
    const open3d::geometry::PointCloud &scene,
    const std::vector<TemplateData> &templates,
    const Eigen::Matrix3d &R_base,
    const Eigen::Vector3d &scene_center,
    int num_planes,
    double icp_dist,
    int yaw_step_deg);

}
