cmake_minimum_required(VERSION 3.16)
project(pcd_block_estimation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------
find_package(Open3D CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0
)
FetchContent_MakeAvailable(yaml-cpp)

# ------------------------------------------------------------
# Library
# ------------------------------------------------------------
add_library(pcd_block_estimation
    src/utils.cpp
    src/pose_estimation.cpp
    src/template_utils.cpp
    src/yaml_utils.cpp
    src/mask_projection.cpp
)

target_include_directories(pcd_block_estimation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(pcd_block_estimation
    PUBLIC
        Open3D::Open3D
        yaml-cpp::yaml-cpp
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

# ------------------------------------------------------------
# Optional standalone tools (OFF by default)
# ------------------------------------------------------------
option(PCD_BLOCK_ESTIMATION_BUILD_TOOLS "Build standalone tools" ON)

if(PCD_BLOCK_ESTIMATION_BUILD_TOOLS)

    add_executable(generate_templates src/main_generate_templates.cpp)
    target_link_libraries(generate_templates PRIVATE pcd_block_estimation)

    add_executable(extract_pcl_by_mask src/main_extract_pcl_by_mask.cpp)
    target_link_libraries(extract_pcl_by_mask PRIVATE pcd_block_estimation)

    add_executable(full_pipeline src/main_full_pipeline.cpp)
    target_link_libraries(full_pipeline PRIVATE pcd_block_estimation)

    add_executable(dump_processing src/main_dump_processing.cpp)
    target_link_libraries(dump_processing PRIVATE pcd_block_estimation)
endif()
