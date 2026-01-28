#include <open3d/Open3D.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

#include <filesystem>
#include <iostream>
#include <map>
#include <vector>
#include <string>

namespace fs = std::filesystem;
using namespace open3d;

// ------------------------------------------------------------
// Defaults (match Python reference)
// ------------------------------------------------------------
struct Params {
    std::string cad_path   = "../data/ConcreteBlock.ply";
    std::string out_dir    = "../data/templates";
    int         n_points   = 2000;
    double      angle_deg  = 15.0;
};

// ------------------------------------------------------------
void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --cad <path>        CAD mesh path (default: ConcreteBlock.ply)\n"
              << "  --out <dir>         Output directory (default: templates)\n"
              << "  --n-points <int>    Number of sampled points (default: 2000)\n"
              << "  --angle-deg <deg>   Face angle threshold in degrees (default: 15.0)\n"
              << "  --help              Show this help message\n";
}

// ------------------------------------------------------------
Params parse_args(int argc, char **argv)
{
    Params p;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--cad" && i + 1 < argc) {
            p.cad_path = argv[++i];
        }
        else if (arg == "--out" && i + 1 < argc) {
            p.out_dir = argv[++i];
        }
        else if (arg == "--n-points" && i + 1 < argc) {
            p.n_points = std::stoi(argv[++i]);
        }
        else if (arg == "--angle-deg" && i + 1 < argc) {
            p.angle_deg = std::stod(argv[++i]);
        }
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    return p;
}

// ------------------------------------------------------------
// Face normals in canonical object frame (OUTWARD)
// ------------------------------------------------------------
const std::map<std::string, Eigen::Vector3d> FACE_NORMALS = {
    {"top",   {0, 0,  1}},
    {"front", {1, 0,  0}},
    {"right", {0, 1,  0}},
    {"left",  {0,-1,  0}},
};

// ------------------------------------------------------------
// Template definitions
// ------------------------------------------------------------
const std::map<std::string, std::vector<std::string>> TEMPLATES = {
    {"top", {"top"}},

    {"top_front_short", {"top", "front"}},
    {"top_front_long",  {"top", "right"}},

    {"top_front_short_left",  {"top", "front", "left"}},
    {"top_front_short_right", {"top", "front", "right"}},
};

// ------------------------------------------------------------
bool point_matches_faces(
    const Eigen::Vector3d &n,
    const std::vector<std::string> &faces,
    double cos_thresh)
{
    for (const auto &f : faces) {
        const auto &fn = FACE_NORMALS.at(f);
        if (n.dot(fn) > cos_thresh)
            return true;
    }
    return false;
}

// ------------------------------------------------------------
int main(int argc, char **argv)
{
    Params params = parse_args(argc, argv);

    fs::create_directories(params.out_dir);

    std::cout << "Generating templates with:\n"
              << "  CAD        : " << params.cad_path << "\n"
              << "  Output dir : " << params.out_dir << "\n"
              << "  N points   : " << params.n_points << "\n"
              << "  Angle deg  : " << params.angle_deg << "\n\n";

    // --------------------------------------------------------
    // Load CAD mesh
    // --------------------------------------------------------
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    if (!io::ReadTriangleMesh(params.cad_path, *mesh) || mesh->IsEmpty()) {
        throw std::runtime_error("Failed to load CAD mesh: " + params.cad_path);
    }

    mesh->ComputeVertexNormals();

    // --------------------------------------------------------
    // Sample point cloud
    // --------------------------------------------------------
    auto pcd = mesh->SamplePointsPoissonDisk(params.n_points);
    if (!pcd || pcd->IsEmpty()) {
        throw std::runtime_error("Poisson disk sampling failed");
    }

    auto &points  = pcd->points_;
    auto &normals = pcd->normals_;

    // --------------------------------------------------------
    // Ensure outward-facing normals
    // --------------------------------------------------------
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (const auto &p : points)
        center += p;
    center /= static_cast<double>(points.size());

    for (size_t i = 0; i < normals.size(); ++i) {
        if (normals[i].dot(points[i] - center) < 0.0) {
            normals[i] *= -1.0;
        }
    }

    // --------------------------------------------------------
    // Face classification
    // --------------------------------------------------------
    const double cos_thresh =
        std::cos(params.angle_deg * M_PI / 180.0);

    // --------------------------------------------------------
    // Generate templates
    // --------------------------------------------------------
    for (const auto &[name, faces] : TEMPLATES) {

        auto tpl = std::make_shared<geometry::PointCloud>();

        for (size_t i = 0; i < points.size(); ++i) {
            if (point_matches_faces(normals[i], faces, cos_thresh)) {
                tpl->points_.push_back(points[i]);
                tpl->normals_.push_back(normals[i]);
            }
        }

        std::cout << name << ": "
                  << tpl->points_.size()
                  << " points\n";

        if (tpl->IsEmpty()) {
            std::cerr << "WARNING: empty template " << name << "\n";
            continue;
        }

        // ----------------------------------------------------
        // Save point clouds
        // ----------------------------------------------------
        io::WritePointCloud(
            params.out_dir + "/" + name + ".pcd", *tpl);

        io::WritePointCloud(
            params.out_dir + "/" + name + ".ply",
            *tpl,
            /*write_ascii=*/false
        );

        // ----------------------------------------------------
        // Metadata
        // ----------------------------------------------------
        YAML::Node meta;
        meta["template_name"] = name;
        meta["num_faces"] = static_cast<int>(faces.size());

        YAML::Node faces_node;
        for (const auto &f : faces) {
            YAML::Node fn;
            fn["name"] = f;

            const auto &n = FACE_NORMALS.at(f);
            fn["normal"] = std::vector<double>{
                n.x(), n.y(), n.z()
            };

            faces_node.push_back(fn);
        }

        meta["faces"] = faces_node;

        std::ofstream fout(params.out_dir + "/" + name + ".yaml");
        fout << meta;
    }

    std::cout << "\nTemplates + metadata written to: "
              << params.out_dir << std::endl;

    return 0;
}
