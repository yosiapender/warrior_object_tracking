#include "detector/yolo_detector.hpp"

#include <opencv2/dnn.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
inline std::vector<int64_t> getShape(const Ort::TensorTypeAndShapeInfo& info) {
    return info.GetShape();
}
inline size_t numel(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d > 0 ? d : 1);
    return n;
}
}

YoloDetector::YoloDetector(const std::string& onnx_path,
                           int input_width,
                           int input_height,
                           float conf_thres,
                           float nms_thres,
                           bool use_cuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "YOLO"),
      input_width_(input_width),
      input_height_(input_height),
      conf_thres_(conf_thres),
      nms_thres_(nms_thres) {
    initSession(onnx_path, use_cuda);
}

void YoloDetector::initSession(const std::string& onnx_path, bool use_cuda) {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[INFO] ONNX Runtime CUDA enabled\n";
        } catch (const std::exception& e) {
            std::cerr << "[WARN] CUDA provider append failed, falling back to CPU. Reason: "
                      << e.what() << "\n";
        }
    } else {
        std::cout << "[INFO] ONNX Runtime CPU\n";
    }

    session_ = Ort::Session(env_, onnx_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;

    const size_t num_inputs = session_.GetInputCount();
    input_names_.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        Ort::AllocatedStringPtr name(session_.GetInputNameAllocated(i, allocator));
        input_names_[i] = name.get();
    }

    const size_t num_outputs = session_.GetOutputCount();
    output_names_.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        Ort::AllocatedStringPtr name(session_.GetOutputNameAllocated(i, allocator));
        output_names_[i] = name.get();
    }

    std::cout << "[INFO] Inputs: " << num_inputs << ", Outputs: " << num_outputs << "\n";
    for (size_t i = 0; i < num_inputs; ++i) std::cout << "  Input[" << i << "]: " << input_names_[i] << "\n";
    for (size_t i = 0; i < num_outputs; ++i) std::cout << "  Output[" << i << "]: " << output_names_[i] << "\n";
}

cv::Mat YoloDetector::preprocess(const cv::Mat& image, float& scale) {
    const int w = image.cols;
    const int h = image.rows;

    scale = std::min(input_width_ / static_cast<float>(w),
                     input_height_ / static_cast<float>(h));

    const int new_w = static_cast<int>(w * scale);
    const int new_h = static_cast<int>(h * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    cv::Mat padded(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(0, 0, new_w, new_h)));

    return cv::dnn::blobFromImage(
        padded,
        1.0 / 255.0,
        cv::Size(input_width_, input_height_),
        cv::Scalar(),
        /*swapRB=*/true,
        /*crop=*/false,
        CV_32F
    );
}

std::vector<Detection> YoloDetector::postprocess(const cv::Size& orig_size,
                                                 float scale,
                                                 const float* out,
                                                 size_t out_count) {
    std::vector<Detection> dets;

    // (1, 5, 8400) => layout [x,y,w,h,conf] each length 8400
    constexpr int NUM_BOXES = 8400;
    constexpr int CHANNELS  = 5;
    const size_t expected = static_cast<size_t>(NUM_BOXES) * CHANNELS;
    if (out_count < expected) {
        std::cerr << "[ERROR] Output tensor too small. Got " << out_count
                  << " floats, expected at least " << expected << "\n";
        return dets;
    }

    const float* xptr = out + 0 * NUM_BOXES;
    const float* yptr = out + 1 * NUM_BOXES;
    const float* wptr = out + 2 * NUM_BOXES;
    const float* hptr = out + 3 * NUM_BOXES;
    const float* cptr = out + 4 * NUM_BOXES;

    dets.reserve(64);

    for (int i = 0; i < NUM_BOXES; ++i) {
        const float conf = cptr[i];
        if (conf < conf_thres_) continue;

        const float cx = xptr[i];
        const float cy = yptr[i];
        const float bw = wptr[i];
        const float bh = hptr[i];

        float x = cx - 0.5f * bw;
        float y = cy - 0.5f * bh;

        x  /= scale; y  /= scale;
        const float ww = bw / scale;
        const float hh = bh / scale;

        const int x0 = std::max(0, static_cast<int>(std::floor(x)));
        const int y0 = std::max(0, static_cast<int>(std::floor(y)));
        const int x1 = std::min(orig_size.width,  static_cast<int>(std::ceil(x + ww)));
        const int y1 = std::min(orig_size.height, static_cast<int>(std::ceil(y + hh)));

        if (x1 <= x0 || y1 <= y0) continue;

        dets.push_back({cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1)), conf, 0});
    }

    return dets;
}

std::vector<Detection> YoloDetector::nms(const std::vector<Detection>& dets) const {
    if (dets.empty()) return {};

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    boxes.reserve(dets.size());
    scores.reserve(dets.size());

    for (const auto& d : dets) {
        boxes.push_back(d.box);
        scores.push_back(d.score);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(
        boxes,
        scores,
        /*score_threshold=*/conf_thres_,
        /*nms_threshold=*/nms_thres_,
        keep
    );

    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int idx : keep) out.push_back(dets[idx]);
    return out;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& image) {
    float scale = 1.0f;
    cv::Mat blob = preprocess(image, scale); // 1x3xH xW float32

    if (blob.empty() || blob.type() != CV_32F) return {};

    const std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    const size_t tensor_size = static_cast<size_t>(1) * 3 * input_height_ * input_width_;
    float* blob_ptr = blob.ptr<float>();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info,
        blob_ptr,
        tensor_size,
        input_shape.data(),
        input_shape.size()
    );

    std::vector<const char*> input_names_c;
    input_names_c.reserve(input_names_.size());
    for (auto& s : input_names_) input_names_c.push_back(s.c_str());

    std::vector<const char*> output_names_c;
    output_names_c.reserve(output_names_.size());
    for (auto& s : output_names_) output_names_c.push_back(s.c_str());

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_c.data(),
        &input_ort,
        1,
        output_names_c.data(),
        output_names_c.size()
    );

    if (outputs.empty()) return {};

    Ort::Value& out0 = outputs[0];
    const float* out_data = out0.GetTensorData<float>();

    const auto type_info = out0.GetTensorTypeAndShapeInfo();
    const size_t out_count = type_info.GetElementCount();

    auto dets = postprocess(image.size(), scale, out_data, out_count);
    return nms(dets);
}
