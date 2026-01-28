#include <open3d/Open3D.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

#include "pcd_block_estimation/utils.hpp"

#include <filesystem>
#include <chrono>
#include <iostream>

using namespace open3d;
using namespace pcd_block;
namespace fs = std::filesystem;

// ------------------------------------------------------------
// Parameters
// ------------------------------------------------------------
static const std::string PCD_PLY = "../data/segmented_concrete_block.ply";
static const std::string TEMPLATE_DIR = "../data/templates";
static const std::string CALIB_YAML = "../data/calib_zed2i_to_seyond.yaml";

constexpr double DIST_THRESH = 0.02;
constexpr int MAX_PLANES = 3;
constexpr int MIN_INLIERS = 100;
constexpr double ICP_DIST = 0.04;

const Eigen::Vector3d Z_WORLD(0.0, -1.0, 0.0); // w.r.t to camera frame
constexpr double ANGLE_THRESH = std::cos(30.0 * M_PI / 180.0);

// ------------------------------------------------------------
int main()
{
    auto t_start = std::chrono::high_resolution_clock::now();

    // --------------------------------------------------------
    // Load point cloud
    // --------------------------------------------------------
    auto pcd = std::make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(PCD_PLY, *pcd) || pcd->IsEmpty())
        throw std::runtime_error("Failed to load point cloud");

    pcd->RemoveStatisticalOutliers(20, 2.0);
    pcd->EstimateNormals();

    // --------------------------------------------------------
    // Load calibration
    // --------------------------------------------------------
    YAML::Node calib = YAML::LoadFile(CALIB_YAML);
    // Eigen::Matrix4d T_lidar_cam;
    // for (int i = 0; i < 16; ++i)
    //     T_lidar_cam(i / 4, i % 4) = calib["T"][i].as<double>();

    Eigen::Matrix4d T_lidar_cam;
    auto Tnode = calib["T"];

    if (!Tnode || !Tnode.IsSequence() || Tnode.size() != 4) {
        throw std::runtime_error("T must be a 4x4 matrix");
    }

    for (int r = 0; r < 4; ++r) {
        if (!Tnode[r].IsSequence() || Tnode[r].size() != 4) {
            throw std::runtime_error("T must be a 4x4 matrix");
        }
        for (int c = 0; c < 4; ++c) {
            T_lidar_cam(r, c) = Tnode[r][c].as<double>();
        }
    }

    Eigen::Matrix4d T_cam_lidar = T_lidar_cam.inverse();

    Eigen::Matrix4d T_pitch = Eigen::Matrix4d::Identity();
    T_pitch.block<3,3>(0,0) =
        geometry::Geometry3D::GetRotationMatrixFromXYZ(
            Eigen::Vector3d(-M_PI / 6.0, 0, 0));

    pcd->Transform(T_pitch * T_cam_lidar);
    pcd->RemoveStatisticalOutliers(20, 2.0);

    Eigen::Vector3d pcd_center = compute_center(*pcd);

    // --------------------------------------------------------
    // Plane extraction
    // --------------------------------------------------------
    auto planes = extract_planes(*pcd, MAX_PLANES,
                                 DIST_THRESH, MIN_INLIERS);

    if (planes.empty())
        throw std::runtime_error("No planes detected");

    std::cout << "nr of planes: " << planes.size() << std::endl;

    Eigen::Vector3d n_top;
    bool found = false;

    for (auto &[plane, pc] : planes)
    {
        Eigen::Vector3d n = plane.head<3>().normalized();
        if (n.dot(Z_WORLD) < 0)
            n = -n;

        if (std::abs(n.dot(Z_WORLD)) > ANGLE_THRESH)
        {
            n_top = n;
            found = true;
            break;
        }
    }

    if (!found)
        throw std::runtime_error("Top plane not detected");

    // --------------------------------------------------------
    // Base frame
    // --------------------------------------------------------
    Eigen::Vector3d z_cam = n_top.normalized();
    Eigen::Vector3d tmp(1, 0, 0);
    if (std::abs(tmp.dot(z_cam)) > 0.9)
        tmp = Eigen::Vector3d(0, 1, 0);

    Eigen::Vector3d x_base =
        (tmp - tmp.dot(z_cam) * z_cam).normalized();
    Eigen::Vector3d y_base = z_cam.cross(x_base).normalized();

    Eigen::Matrix3d R_base;
    R_base.col(0) = x_base;
    R_base.col(1) = y_base;
    R_base.col(2) = z_cam;

    // --------------------------------------------------------
    // Load templates
    // --------------------------------------------------------
    struct Template
    {
        std::string name;
        std::shared_ptr<geometry::PointCloud> pcd;
        int num_faces;
    };

    std::vector<Template> templates;

    for (const auto &f : fs::directory_iterator(TEMPLATE_DIR))
    {
        if (f.path().extension() != ".ply")
            continue;

        auto yaml_path = f.path();
        yaml_path.replace_extension(".yaml");

        if (!fs::exists(yaml_path))
            continue;

        YAML::Node meta = YAML::LoadFile(yaml_path.string());

        auto tpl = std::make_shared<geometry::PointCloud>();
        io::ReadPointCloud(f.path().string(), *tpl);
        tpl->EstimateNormals();

        templates.push_back({
            f.path().filename().string(),
            tpl,
            meta["num_faces"].as<int>()
        });
    }

    // --------------------------------------------------------
    // ICP sweep
    // --------------------------------------------------------
    pipelines::registration::RegistrationResult best;
    std::string best_name;
    int best_yaw = 0;
    int best_index = 0;

    int tpl_index = 0;
    for (const auto &tpl : templates)
    {
        if (tpl.num_faces != static_cast<int>(planes.size()))
            continue;

        std::cout << "pcl nr of planes: " << planes.size() << ", template "<< tpl.name << ", nr of planes: " << tpl.num_faces << std::endl;

        Eigen::Vector3d tpl_center = compute_center(*tpl.pcd);

        for (int yaw = 0; yaw < 360; yaw += 30)
        {
            Eigen::Matrix3d R_yaw =
                geometry::Geometry3D::GetRotationMatrixFromAxisAngle(
                    z_cam * yaw * M_PI / 180.0);

            Eigen::Matrix3d R_init = R_yaw * R_base;

            Eigen::Matrix4d T_init = Eigen::Matrix4d::Identity();
            T_init.block<3,3>(0,0) = R_init;
            T_init.block<3,1>(0,3) =
                pcd_center - R_init * tpl_center;

            auto result =
                pipelines::registration::RegistrationICP(
                    *tpl.pcd, *pcd,
                    ICP_DIST,
                    T_init,
                    pipelines::registration::
                        TransformationEstimationPointToPlane()
                );

            if (result.fitness_ > best.fitness_)
            {
                best = result;
                best_name = tpl.name;
                best_yaw = yaw;
                best_index = tpl_index;
            }
        }
        tpl_index++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Best template: " << best_name << "\n";
    std::cout << "Best yaw: " << best_yaw << " deg\n";
    std::cout << "Best index: " << best_index << "\n";
    std::cout << "Fitness=" << best.fitness_
              << ", RMSE=" << best.inlier_rmse_ << "\n";

    std::cout << "Execution time: "
              << std::chrono::duration<double, std::milli>(
                     t_end - t_start).count()
              << " ms\n";

    // --------------------------------------------------------
    // Visualization
    // --------------------------------------------------------
    auto best_vis = std::make_shared<geometry::PointCloud>(*templates[best_index].pcd);
    best_vis->Transform(best.transformation_);
    best_vis->PaintUniformColor({0,1,0});

    auto frame =
        geometry::TriangleMesh::CreateCoordinateFrame(0.3);
    frame->Transform(best.transformation_);

    visualization::DrawGeometries(
        {pcd, best_vis, frame},
        "Final Pose Estimation Result",
        1200, 900
    );

    return 0;
}
