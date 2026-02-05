#include "pcd_block_estimation/template_utils.hpp"

#include <iostream>

using namespace pcd_block;

// ------------------------------------------------------------
void print_usage(const char * prog)
{
  std::cout
    << "Usage: " << prog << " [options]\n\n"
    << "Options:\n"
    << "  --cad <path>\n"
    << "  --out <dir>\n"
    << "  --n-points <int>\n"
    << "  --angle-deg <deg>\n";
}

// ------------------------------------------------------------
TemplateGenerationParams parse_args(int argc, char ** argv)
{
  TemplateGenerationParams p;
  p.cad_path = "../data/ConcreteBlock.ply";
  p.out_dir = "../data/templates";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--cad" && i + 1 < argc) {
      p.cad_path = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      p.out_dir = argv[++i];
    } else if (arg == "--n-points" && i + 1 < argc) {
      p.n_points = std::stoi(argv[++i]);
    } else if (arg == "--angle-deg" && i + 1 < argc) {
      p.angle_deg = std::stod(argv[++i]);
    } else {
      print_usage(argv[0]);
      std::exit(1);
    }
  }
  return p;
}

// ------------------------------------------------------------
int main(int argc, char ** argv)
{
  auto params = parse_args(argc, argv);

  std::cout << "Generating templates from "
            << params.cad_path << std::endl;

  generate_templates(params);

  std::cout << "Done. Output: "
            << params.out_dir << std::endl;
  return 0;
}
