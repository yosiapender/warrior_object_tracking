# Warrior Object Tracking
A real-time object tracking system combining YOLO detection with OpenCV tracking algorithms.

## Features
- YOLO-based object detection using ONNX models
- Multiple tracker support (CSRT, KCF)
- Video processing and evaluation tools
- Modular C++ architecture

## Project Structure
warrior_object_tracking/
├── data/ # Dataset and output directories
├── include/ # Header files
├── models/ # ONNX model files
├── src/ # Source code
├── tools/ # Python utilities
└── CMakeLists.txt # Build configuration


## Quick Start

### Prerequisites
- Ubuntu 20.04/22.04
- OpenCV 4.x
- CMake 3.16+
- C++17 compiler

### Build Instructions
```bash
git clone https://github.com/yourusername/warrior_object_tracking.git
cd warrior_object_tracking

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)

# Process a video file
./main_video ../data/videos/input.mp4 ../models/best.onnx

# Run evaluation
./main_eval ../data/ ../models/best.onnx