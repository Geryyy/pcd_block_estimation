#pragma once
#include <open3d/Open3D.h>
#include <string>
#include <vector>

namespace pcd_block {

struct TemplateData {
    std::string name;
    std::shared_ptr<open3d::geometry::PointCloud> pcd;
    int num_faces;
};

std::vector<TemplateData>
load_templates(const std::string &directory);

}
