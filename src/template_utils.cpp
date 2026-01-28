#include "pcd_block_estimation/template_utils.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace fs = std::filesystem;
using namespace open3d;

namespace pcd_block {

std::vector<TemplateData>
load_templates(const std::string &dir)
{
    std::vector<TemplateData> out;

    for (const auto &f : fs::directory_iterator(dir)) {
        if (f.path().extension() != ".ply") continue;

        auto yaml = f.path();
        yaml.replace_extension(".yaml");
        if (!fs::exists(yaml)) continue;

        YAML::Node meta = YAML::LoadFile(yaml.string());

        auto pcd = std::make_shared<geometry::PointCloud>();
        io::ReadPointCloud(f.path().string(), *pcd);
        pcd->EstimateNormals();

        out.push_back({
            f.path().filename().string(),
            pcd,
            meta["num_faces"].as<int>()
        });
    }
    return out;
}

}
