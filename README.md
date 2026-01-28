# pcd_block_estimation

## Dependencies

`sudo apt install libopen3d-dev libyaml-cpp-dev libeigen3-dev`

```bash
sudo apt update
sudo apt install libglu1-mesa-dev mesa-common-dev
```

## Usage

## build
```bash
mkdir build && cd build
cmake ..
make
```

### Generate templates
```bash
cd build
./generate_templates
```
### execute block pose estimation
```bash
cd build
./pcd_pose_estimator
```