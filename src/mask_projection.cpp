#include "pcd_block_estimation/mask_projection.hpp"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace pcd_block {

Eigen::Vector2i mask_center_uv(const cv::Mat &mask)
{
    cv::Moments m = cv::moments(mask, true);
    if (m.m00 == 0.0)
        throw std::runtime_error("Mask is empty");

    return Eigen::Vector2i(
        static_cast<int>(m.m10 / m.m00),
        static_cast<int>(m.m01 / m.m00)
    );
}

std::vector<Eigen::Vector3d>
select_points_by_mask(
    const std::vector<Eigen::Vector3d> &pts,
    const cv::Mat &mask,
    const Eigen::Matrix3d &K,
    const Eigen::Matrix4d &T_P_C,
    double z_min)
{
    const int H = mask.rows;
    const int W = mask.cols;

    // Inverse: P → C for projection
    Eigen::Matrix4d T_C_P = T_P_C.inverse();
    Eigen::Matrix3d R = T_C_P.block<3,3>(0,0);
    Eigen::Vector3d t = T_C_P.block<3,1>(0,3);

    std::vector<Eigen::Vector3d> selected;
    selected.reserve(pts.size() / 10);

    for (const auto &p : pts) {
        // Point cloud → camera
        Eigen::Vector3d Xc = R * p + t;
        if (Xc.z() <= z_min)
            continue;

        // Pinhole projection
        int u = static_cast<int>(
            std::round(K(0,0) * Xc.x() / Xc.z() + K(0,2)));
        int v = static_cast<int>(
            std::round(K(1,1) * Xc.y() / Xc.z() + K(1,2)));

        if (u < 0 || u >= W || v < 0 || v >= H)
            continue;

        if (mask.at<uint8_t>(v, u) > 0)
            selected.push_back(p);   // keep original frame
    }

    return selected;
}

} // namespace pcd_block
