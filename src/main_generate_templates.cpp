#include "pcd_block_estimation/template_utils.hpp"

#include <iostream>
#include <string>

using namespace pcd_block;

// ------------------------------------------------------------
static void print_usage(const char * prog)
{
  std::cout
    << "Usage: " << prog << " [options]\n\n"
    << "Options:\n"
    << "  --cad <path>        Path to CAD mesh (canonical frame)\n"
    << "  --out <dir>         Output directory for templates\n"
    << "  --n-points <int>    Number of sampled points (default: 2000)\n"
    << "  --angle-deg <deg>   Normal angle threshold for face extraction\n\n"
    << "Notes:\n"
    << "  CAD frame assumptions:\n"
    << "    +Z : top face normal\n"
    << "    +X : front face (short side)\n"
    << "    +Y : long side\n";
}

// ------------------------------------------------------------
static TemplateGenerationParams parse_args(int argc, char ** argv)
{
  TemplateGenerationParams p;
  p.cad_path  = "../data/ConcreteBlock.ply";
  p.out_dir   = "../data/templates";
  p.n_points  = 2000;
  p.angle_deg = 15.0;

  for (int i = 1; i < argc; ++i) {

    std::string arg = argv[i];

    if (arg == "--cad" && i + 1 < argc) {
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

  std::cout << "Generating templates\n"
            << "  CAD path : " << params.cad_path << "\n"
            << "  Output   : " << params.out_dir << "\n"
            << "  Points   : " << params.n_points << "\n"
            << "  Angle    : " << params.angle_deg << " deg\n\n"
            << "Assumed CAD frame:\n"
            << "  top   = +Z\n"
            << "  front = +X (short side)\n"
            << "  side  = +Y (long side)\n"
            << std::endl;

  generate_templates(params);

  std::cout << "\nTemplate generation finished.\n"
            << "Templates written to: "
            << params.out_dir << std::endl;

  return 0;
}
