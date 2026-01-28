#pragma once
#include <Eigen/Dense>
#include <string>

namespace pcd_block {

Eigen::Matrix4d load_T_4x4(const std::string &yaml_path);

}
