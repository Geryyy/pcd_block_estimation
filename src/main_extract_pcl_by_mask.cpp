#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <iostream>

#include "pcd_block_estimation/yaml_utils.hpp"
#include "pcd_block_estimation/utils.hpp"
#include "pcd_block_estimation/mask_projection.hpp"

using namespace open3d;
using namespace pcd_block;

int main()
{
    const std::string PCL_PATH   = "../data/pcl.ply";
    const std::string MASK_PATH  = "../data/mask.png";
    const std::string CALIB_YAML = "../data/calib_zed2i_to_seyond.yaml";

    // --------------------------------------------------------
    // Load mask + center
    // --------------------------------------------------------
    cv::Mat mask = cv::imread(MASK_PATH, cv::IMREAD_GRAYSCALE);
    if (mask.empty())
        throw std::runtime_error("Failed to load mask");

    Eigen::Vector2i uv_center = mask_center_uv(mask);
    std::cout << "Mask center (u,v): "
              << uv_center.transpose() << std::endl;

    // --------------------------------------------------------
    // Load calibration
    // --------------------------------------------------------
    YAML::Node calib = YAML::LoadFile(CALIB_YAML);

    Eigen::Matrix3d K;
    for (int i = 0; i < 9; ++i)
        K(i/3, i%3) = calib["K"][i].as<double>();

    Eigen::Matrix4d T_P_C = load_T_4x4(CALIB_YAML);

    // --------------------------------------------------------
    // Load point cloud
    // --------------------------------------------------------
    auto pcd = std::make_shared<geometry::PointCloud>();
    io::ReadPointCloud(PCL_PATH, *pcd);

    std::cout << "Loaded " << pcd->points_.size()
              << " points\n";

    // --------------------------------------------------------
    // Mask-based cutout
    // --------------------------------------------------------
    auto pts_sel = select_points_by_mask(
        pcd->points_, mask, K, T_P_C);

    std::cout << "Selected " << pts_sel.size()
              << " points\n";

    Eigen::Vector3d center = compute_center(pts_sel);
    std::cout << "Estimated center (x,y,z): "
              << center.transpose() << std::endl;

    // --------------------------------------------------------
    // Save
    // --------------------------------------------------------
    geometry::PointCloud out;
    out.points_ = pts_sel;
    io::WritePointCloud("../data/segmented.ply", out);

    return 0;
}
