cmake_minimum_required(VERSION 3.16)
project(pcd_pose_estimation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# ------------------------------------------------------------
# Build type (default: Release)
# ------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build (Release, Debug, RelWithDebInfo, MinSizeRel)"
        FORCE)
endif()

# ------------------------------------------------------------
# Native CPU optimizations (Release only)
# ------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
        -O3
        -march=native
        # -ffast-math
    )
endif()

# ------------------------------------------------------------
# Open3D binary location
# ------------------------------------------------------------
# set(Open3D_ROOT
#     "/home/gebmer/libs/open3d-devel-linux-x86_64-cxx11-abi-0.19.0"
# )

# Important: make sure CMake finds Open3DConfig.cmake
# list(APPEND CMAKE_PREFIX_PATH "${Open3D_ROOT}")

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------
find_package(Open3D CONFIG REQUIRED)
# find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)


message(STATUS "Found Open3D version: ${Open3D_VERSION}")
message(STATUS "Open3D include dirs: ${Open3D_INCLUDE_DIRS}")

# ------------------------------------------------------------
# Library
# ------------------------------------------------------------
add_library(pcd_pose_utils
    src/utils.cpp
    src/pose_estimation.cpp
    src/template_utils.cpp
    src/yaml_utils.cpp
    src/mask_projection.cpp
)

target_include_directories(pcd_pose_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(pcd_pose_utils
    PUBLIC
        Open3D::Open3D
        yaml-cpp::yaml-cpp
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

# ------------------------------------------------------------
# Executable
# ------------------------------------------------------------
add_executable(pcd_pose_estimator
    src/main_pose_estimator.cpp
)

target_link_libraries(pcd_pose_estimator
    PRIVATE
        pcd_pose_utils
)

# ------------------------------------------------------------
# generate templates
# ------------------------------------------------------------

add_executable(generate_templates
    src/main_generate_templates.cpp
)

target_link_libraries(generate_templates
    PRIVATE
        Open3D::Open3D
        yaml-cpp::yaml-cpp
        Eigen3::Eigen
)

set_target_properties(generate_templates PROPERTIES
    BUILD_RPATH "${Open3D_ROOT}/lib"
    INSTALL_RPATH "${Open3D_ROOT}/lib"
)


# ------------------------------------------------------------
# extrat pcl by mask
# ------------------------------------------------------------

add_executable(extract_pcl_by_mask
    src/main_extract_pcl_by_mask.cpp
)

target_link_libraries(extract_pcl_by_mask
    PRIVATE
        Open3D::Open3D
        yaml-cpp::yaml-cpp
        ${OpenCV_LIBS}
        pcd_pose_utils
)

target_include_directories(extract_pcl_by_mask
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

# ------------------------------------------------------------
# full pipeline: extract pcl and pose estimation
# ------------------------------------------------------------

add_executable(full_pipeline
    src/main_full_pipeline.cpp
)

target_link_libraries(full_pipeline
    PRIVATE
        Open3D::Open3D
        yaml-cpp::yaml-cpp
        ${OpenCV_LIBS}
        pcd_pose_utils
)

target_include_directories(full_pipeline
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

# ------------------------------------------------------------
# RPATH (important for binary Open3D)
# ------------------------------------------------------------
set_target_properties(pcd_pose_estimator PROPERTIES
    BUILD_RPATH "${Open3D_ROOT}/lib"
    INSTALL_RPATH "${Open3D_ROOT}/lib"
)

# ------------------------------------------------------------
# Install (ROS2-compatible layout)
# ------------------------------------------------------------
install(TARGETS pcd_pose_estimator pcd_pose_utils
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)
