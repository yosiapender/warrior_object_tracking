#include "detector/yolo_detector.hpp"

#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>

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

    // Input names
    size_t num_inputs = session_.GetInputCount();
    input_names_.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        Ort::AllocatedStringPtr name(session_.GetInputNameAllocated(i, allocator));
        input_names_[i] = name.get();
    }

    // Output names
    size_t num_outputs = session_.GetOutputCount();
    output_names_.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        Ort::AllocatedStringPtr name(session_.GetOutputNameAllocated(i, allocator));
        output_names_[i] = name.get();
    }

    // Optional: print I/O info once
    std::cout << "[INFO] Inputs: " << num_inputs << ", Outputs: " << num_outputs << "\n";
    for (size_t i = 0; i < num_inputs; ++i) std::cout << "  Input[" << i << "]: " << input_names_[i] << "\n";
    for (size_t i = 0; i < num_outputs; ++i) std::cout << "  Output[" << i << "]: " << output_names_[i] << "\n";
}

cv::Mat YoloDetector::preprocess(const cv::Mat& image, float& scale) {
    int w = image.cols;
    int h = image.rows;

    // letterbox scale (top-left placement like your original)
    scale = std::min(input_width_ / (float)w, input_height_ / (float)h);

    int new_w = int(w * scale);
    int new_h = int(h * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    cv::Mat padded(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(0, 0, new_w, new_h)));

    // Convert to NCHW float32, normalized, RGB
    // blob shape: 1x3xH xW
    cv::Mat blob = cv::dnn::blobFromImage(
        padded,
        1.0 / 255.0,
        cv::Size(input_width_, input_height_),
        cv::Scalar(),
        /*swapRB=*/true,
        /*crop=*/false,
        CV_32F
    );
    return blob;
}

std::vector<Detection> YoloDetector::postprocess(const cv::Size& orig_size,
                                                 float scale,
                                                 const float* out,
                                                 size_t output_count) {
    std::vector<Detection> dets;

    // Expect YOLO export: (1, 5, 8400) => total 42000 floats
    // If model differs, we fail safely instead of drawing nonsense.
    constexpr int NUM_BOXES = 8400;
    constexpr int CHANNELS = 5;
    const size_t expected = (size_t)NUM_BOXES * CHANNELS;

    if (output_count < expected) {
        std::cerr << "[ERROR] Output tensor too small. Got " << output_count
                  << " floats, expected at least " << expected << "\n";
        return dets;
    }

    // Layout: [x(8400), y(8400), w(8400), h(8400), conf(8400)]
    const float* xptr = out + 0 * NUM_BOXES;
    const float* yptr = out + 1 * NUM_BOXES;
    const float* wptr = out + 2 * NUM_BOXES;
    const float* hptr = out + 3 * NUM_BOXES;
    const float* cptr = out + 4 * NUM_BOXES;

    dets.reserve(64);

    for (int i = 0; i < NUM_BOXES; ++i) {
        float conf = cptr[i];
        if (conf < conf_thres_) continue;

        float cx = xptr[i];
        float cy = yptr[i];
        float bw = wptr[i];
        float bh = hptr[i];

        float x = cx - 0.5f * bw;
        float y = cy - 0.5f * bh;

        // Your preprocess places resized at (0,0) with padding on right/bottom.
        // So no pad subtraction needed. Just scale back.
        x  /= scale;
        y  /= scale;
        bw /= scale;
        bh /= scale;

        int x0 = std::max(0, (int)std::floor(x));
        int y0 = std::max(0, (int)std::floor(y));
        int x1 = std::min(orig_size.width,  (int)std::ceil(x + bw));
        int y1 = std::min(orig_size.height, (int)std::ceil(y + bh));

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
        /*score_threshold=*/conf_thres_,  // can also be 0.0f
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

    // Build input tensor
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    const size_t tensor_size =
        (size_t)input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];

    std::vector<float> input_tensor(tensor_size);
    std::memcpy(input_tensor.data(), blob.ptr<float>(), tensor_size * sizeof(float));

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor.data(),
        input_tensor.size(),
        input_shape.data(),
        input_shape.size()
    );

    // C-string names
    std::vector<const char*> input_names_c;
    input_names_c.reserve(input_names_.size());
    for (auto& s : input_names_) input_names_c.push_back(s.c_str());

    std::vector<const char*> output_names_c;
    output_names_c.reserve(output_names_.size());
    for (auto& s : output_names_) output_names_c.push_back(s.c_str());

    // Run
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_c.data(),
        &input_ort,
        1,
        output_names_c.data(),
        output_names_c.size()
    );

    if (outputs.empty()) return {};

    // Get output buffer and count
    Ort::Value& out0 = outputs[0];
    float* out_data = out0.GetTensorMutableData<float>();

    auto type_info = out0.GetTensorTypeAndShapeInfo();
    size_t out_count = type_info.GetElementCount();

    auto dets = postprocess(image.size(), scale, out_data, out_count);
    return nms(dets);
}