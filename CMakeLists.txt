cmake_minimum_required(VERSION 3.18)
project(object_tracking LANGUAGES CXX CUDA)

# =========================
# Compiler standard
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# CUDA (NOTE: ONNX Runtime CUDA TIDAK MEMERLUKAN nvcc)
# =========================
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 2050 (SM 8.6)

# =========================
# OpenCV (IO, drawing only)
# =========================
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

# =========================
# ONNX Runtime (installed via make install)
# =========================
find_package(onnxruntime REQUIRED)

# =========================
# Include directories
# =========================
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# =========================
# Common sources (shared by both executables)
# =========================
set(COMMON_SOURCES
    src/detector/yolo_detector.cpp
    src/tracker/kcf_tracker.cpp
    src/tracker/csrt_tracker.cpp
    src/tracker/kcf_local.cpp
    src/tracker/colornames_lut.cpp
)

# =========================
# Executables
# =========================

# Live camera executable (your existing main)
add_executable(object_tracking
    src/main.cpp
    ${COMMON_SOURCES}
)

# Video file executable (NEW)
add_executable(object_tracking_video
    src/main_video.cpp
    ${COMMON_SOURCES}
)

# Video file executable (NEW)
add_executable(object_tracking_eval
    src/main_eval.cpp
    ${COMMON_SOURCES}
)


# =========================
# Link libraries (both)
# =========================
target_link_libraries(object_tracking
    PRIVATE
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
)

target_link_libraries(object_tracking_video
    PRIVATE
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
)

target_link_libraries(object_tracking_eval
    PRIVATE
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
)

# =========================
# Compile definitions (both)
# =========================
target_compile_definitions(object_tracking
    PRIVATE USE_ONNXRUNTIME
)

target_compile_definitions(object_tracking_video
    PRIVATE USE_ONNXRUNTIME
)

target_compile_definitions(object_tracking_eval
    PRIVATE USE_ONNXRUNTIME
)

# =========================
# Output (both)
# =========================
set_target_properties(object_tracking PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

set_target_properties(object_tracking_video PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

set_target_properties(object_tracking_eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)