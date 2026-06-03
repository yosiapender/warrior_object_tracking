#pragma once

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>

namespace app {

class Timer {
public:
    void start() { t1_ = std::chrono::high_resolution_clock::now(); }
    double stopMs() const {
        const auto t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point t1_{};
};

struct Policy {
    int   redetect_every = 15;
    float init_expand    = 1.25f;
    float yolo_min_conf  = 0.25f;

    bool   snap_on_periodic = true;
    double snap_center_px   = 30.0;
    double snap_iou_min     = 0.20;

    double max_center_jump_px = 120.0;
    double min_area_px        = 16.0;
    double max_area_frac      = 0.35;
    double min_inframe_frac   = 0.65;
};

inline cv::Rect clampRect(const cv::Rect& r, const cv::Size& sz) {
    return r & cv::Rect(0, 0, sz.width, sz.height);
}

inline cv::Point2f rectCenter(const cv::Rect& r) {
    return cv::Point2f(r.x + r.width * 0.5f, r.y + r.height * 0.5f);
}

inline double IoU(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect inter = a & b;
    const double interA = static_cast<double>(inter.area());
    const double unionA = static_cast<double>(a.area()) + static_cast<double>(b.area()) - interA;
    return (unionA > 0.0) ? (interA / unionA) : 0.0;
}

inline cv::Rect expandBox(const cv::Rect& b, float scale, const cv::Size& sz) {
    const cv::Point2f c = rectCenter(b);
    const int nw = std::max(2, static_cast<int>(std::lround(b.width  * scale)));
    const int nh = std::max(2, static_cast<int>(std::lround(b.height * scale)));
    const int x  = static_cast<int>(std::lround(c.x - nw * 0.5f));
    const int y  = static_cast<int>(std::lround(c.y - nh * 0.5f));
    return clampRect(cv::Rect(x, y, nw, nh), sz);
}

inline cv::Rect shrinkBox(const cv::Rect& b, float inv_scale, const cv::Size& sz) {
    return expandBox(b, inv_scale, sz);
}

inline bool validBox(const cv::Rect& r, const cv::Size& sz, const Policy& p) {
    if (r.width <= 1 || r.height <= 1) return false;

    const double area = static_cast<double>(r.area());
    if (area < p.min_area_px) return false;

    const double frame_area = static_cast<double>(sz.area());
    if (area > p.max_area_frac * frame_area) return false;

    const cv::Rect cl = clampRect(r, sz);
    const double in_frac = (area > 0.0) ? (static_cast<double>(cl.area()) / area) : 0.0;
    if (in_frac < p.min_inframe_frac) return false;

    return true;
}

inline void drawTopRightStatus(cv::Mat& frame,
                              const std::string& text,
                              int line_idx,
                              double scale = 0.8,
                              int thickness = 2,
                              int margin = 10,
                              int line_gap = 8) {
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                        scale, thickness, &baseline);
    const int x = frame.cols - margin - ts.width;
    const int y = margin + ts.height + line_idx * (ts.height + line_gap);
    cv::putText(frame, text, {x, y},
                cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
}
}