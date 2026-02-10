#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

class KcfTracker {
public:
    KcfTracker();  // now uses KCF(CN) internally

    // Initialize tracker with first frame + initial box
    bool init(const cv::Mat& frame, const cv::Rect& init_box);

    // Update tracker on new frame
    bool update(const cv::Mat& frame, cv::Rect& out_box);

    void reset();
    bool isInitialized() const { return initialized_; }

private:
    cv::Ptr<cv::Tracker> tracker_;
    bool initialized_ = false;

    // Store params so you can tune later if needed
    cv::TrackerKCF::Params params_;
};
