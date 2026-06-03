#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

class CsrtTracker {
public:
    CsrtTracker();

    bool init(const cv::Mat& frame, const cv::Rect& init_box);

    bool update(const cv::Mat& frame, cv::Rect& out_box);

    void reset();
    bool isInitialized() const { return initialized_; }

private:
    cv::Ptr<cv::Tracker> tracker_;
    bool initialized_ = false;
};
