#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <stdexcept>

using namespace open3d;

// ------------------------------------------------------------
// Helper: load 4x4 matrix from YAML
// ------------------------------------------------------------
Eigen::Matrix4d load_T(const std::string &path)
{
    YAML::Node root = YAML::LoadFile(path);
    auto Tn = root["T"];

    if (!Tn || !Tn.IsSequence() || Tn.size() != 4)
        throw std::runtime_error("T must be 4x4");

    Eigen::Matrix4d T;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            T(r,c) = Tn[r][c].as<double>();
    return T;
}

// ------------------------------------------------------------
// Project + mask filtering
// ------------------------------------------------------------

/**
 * @brief Select points from a point cloud whose image projections lie inside a binary mask.
 *
 * The input points are given in the POINT CLOUD frame (e.g. LiDAR frame).
 * Points are TEMPORARILY transformed into the CAMERA frame for projection,
 * but all returned points remain in the ORIGINAL point cloud frame.
 *
 * Coordinate frames:
 *   - P : point cloud frame (LiDAR / world frame of pts)
 *   - C : camera optical frame
 *
 * @param pts
 *   3D points in the point cloud frame P.
 *   Each point is p_i = (x, y, z)^T in frame P.
 *
 * @param mask
 *   Binary image mask (H×W).
 *   Non-zero values indicate valid pixels.
 *   Pixel coordinates follow OpenCV convention:
 *   u = column (x), v = row (y).
 *
 * @param K
 *   Camera intrinsic matrix (3×3):
 *
 *       [ fx  0  cx ]
 *       [  0 fy  cy ]
 *       [  0  0   1 ]
 *
 * @param D
 *   Camera distortion coefficients (optional).
 *   If empty, distortion is ignored.
 *
 * @param T
 *   Homogeneous extrinsic transformation from POINT CLOUD frame to CAMERA frame:
 *
 *       T = T_{P→C}
 *
 *   This means:
 *
 *       X_C = R_{PC} * X_P + t_{PC}
 *
 *   where:
 *     - X_P is a point in the point cloud frame P
 *     - X_C is the same point expressed in the camera frame C
 *
 * @param z_min
 *   Minimum depth in camera frame.
 *   Points with z_C <= z_min are discarded before projection.
 *
 * @return
 *   Vector of selected points in the ORIGINAL point cloud frame P.
 *   No frame conversion is applied to the returned points.
 *
 * Notes:
 *   Internally, the inverse transform is used:
 *
 *       T_inv = T_{C←P} = T_{P→C}^{-1}
 *
 *   This inverse is applied only for projection into the image.
 *   The stored points remain in frame P.
 */
std::vector<Eigen::Vector3d>
select_points_by_mask(
    const std::vector<Eigen::Vector3d> &pts,
    const cv::Mat &mask,
    const Eigen::Matrix3d &K,
    const Eigen::VectorXd &D,
    const Eigen::Matrix4d &T,
    double z_min = 0.1)
{
    const int H = mask.rows;
    const int W = mask.cols;

    Eigen::Matrix4d Tinv = T.inverse();
    Eigen::Matrix3d R = Tinv.block<3,3>(0,0);
    Eigen::Vector3d t = Tinv.block<3,1>(0,3);

    std::vector<Eigen::Vector3d> selected;

    for (const auto &p : pts) {
        Eigen::Vector3d Xc = R * p + t;
        if (Xc.z() < z_min)
            continue;

        Eigen::Vector2d uv;
        uv.x() = K(0,0) * Xc.x() / Xc.z() + K(0,2);
        uv.y() = K(1,1) * Xc.y() / Xc.z() + K(1,2);

        int u = static_cast<int>(std::round(uv.x()));
        int v = static_cast<int>(std::round(uv.y()));

        if (u < 0 || u >= W || v < 0 || v >= H)
            continue;

        if (mask.at<uint8_t>(v, u) > 0)
            selected.push_back(p);
    }

    return selected;
}

// ------------------------------------------------------------
int main()
{
    const std::string PCL_PATH   = "../data/pcl.ply";
    const std::string MASK_PATH  = "../data/mask.png";
    const std::string CALIB_YAML = "../data/calib_zed2i_to_seyond.yaml";

    // --------------------------------------------------------
    // Load mask
    // --------------------------------------------------------
    cv::Mat mask = cv::imread(MASK_PATH, cv::IMREAD_GRAYSCALE);
    if (mask.empty())
        throw std::runtime_error("Failed to load mask");

    // --------------------------------------------------------
    // Compute mask center (image space)
    // --------------------------------------------------------
    cv::Moments m = cv::moments(mask, true);
    if (m.m00 == 0)
        throw std::runtime_error("Empty mask");

    int cx_img = static_cast<int>(m.m10 / m.m00);
    int cy_img = static_cast<int>(m.m01 / m.m00);

    std::cout << "Mask center (u,v): "
              << cx_img << ", " << cy_img << std::endl;

    // --------------------------------------------------------
    // Load calibration
    // --------------------------------------------------------
    YAML::Node calib = YAML::LoadFile(CALIB_YAML);

    Eigen::Matrix3d K;
    for (int i = 0; i < 9; ++i)
        K(i/3, i%3) = calib["K"][i].as<double>();

    Eigen::VectorXd D;
    if (calib["D"])
        D = Eigen::VectorXd::Map(
            calib["D"].as<std::vector<double>>().data(),
            calib["D"].size());

    Eigen::Matrix4d T = load_T(CALIB_YAML);

    // --------------------------------------------------------
    // Load point cloud
    // --------------------------------------------------------
    auto pcd = std::make_shared<geometry::PointCloud>();
    io::ReadPointCloud(PCL_PATH, *pcd);

    std::vector<Eigen::Vector3d> pts = pcd->points_;
    std::cout << "Loaded " << pts.size() << " points\n";

    // --------------------------------------------------------
    // Extract masked points
    // --------------------------------------------------------
    auto pts_sel = select_points_by_mask(
        pts, mask, K, D, T);

    std::cout << "Selected " << pts_sel.size()
              << " points inside mask\n";

    // --------------------------------------------------------
    // Estimate 3D center from selected points
    // --------------------------------------------------------
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (const auto &p : pts_sel)
        center += p;
    center /= pts_sel.size();

    std::cout << "Estimated object center (x,y,z): "
              << center.transpose() << std::endl;

    // --------------------------------------------------------
    // Save cut-out cloud
    // --------------------------------------------------------
    geometry::PointCloud pcd_out;
    pcd_out.points_ = pts_sel;

    io::WritePointCloud("../data/segmented.ply", pcd_out);

    return 0;
}
