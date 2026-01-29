#include "pcd_block_estimation/yaml_utils.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace pcd_block
{

Eigen::Matrix4d load_T_4x4(const std::string & path)
{
  YAML::Node root = YAML::LoadFile(path);
  auto Tnode = root["T"];

  if (!Tnode || !Tnode.IsSequence() || Tnode.size() != 4) {
    throw std::runtime_error("T must be 4x4");
  }

  Eigen::Matrix4d T;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      T(r, c) = Tnode[r][c].as<double>();
    }
  }

  return T;
}

Eigen::Matrix3d load_camera_matrix(const std::string & path)
{
  YAML::Node root = YAML::LoadFile(path);
  Eigen::Matrix3d K;
  for (int i = 0; i < 9; ++i) {
    K(i / 3, i % 3) = root["K"][i].as<double>();
  }
  return K;
}

}
