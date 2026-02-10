#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

class YoloDetector {
public:
    YoloDetector(const std::string& onnx_path,
                 int input_width = 640,
                 int input_height = 640,
                 float conf_thres = 0.25f,
                 float nms_thres  = 0.45f,
                 bool use_cuda = true);

    std::vector<Detection> detect(const cv::Mat& image);

private:
    void initSession(const std::string& onnx_path, bool use_cuda);

    // Returns NCHW float32 blob (1x3xH xW), normalized [0,1], RGB
    cv::Mat preprocess(const cv::Mat& image, float& scale);

    std::vector<Detection> postprocess(const cv::Size& orig_size,
                                       float scale,
                                       const float* output,
                                       size_t output_count);

    std::vector<Detection> nms(const std::vector<Detection>& dets) const;

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::SessionOptions session_options_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    int input_width_;
    int input_height_;
    float conf_thres_;
    float nms_thres_;
};