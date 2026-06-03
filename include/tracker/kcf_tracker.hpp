#pragma once

#include <opencv2/core.hpp>
#include <memory>

#include "tracker/kcf_local.hpp"

class KcfTracker {
public:
    KcfTracker();

    bool init(const cv::Mat& frame, const cv::Rect& init_box);
    bool update(const cv::Mat& frame, cv::Rect& out_box);
    bool getSearchWindowRect(cv::Rect& out) const;

    bool getLastResponseMap(cv::Mat& out_response32f) const;

    void reset();
    bool isInitialized() const { return initialized_; }

private:
    std::unique_ptr<KcfLocalTracker> tracker_;
    bool initialized_ = false;
    KcfLocalTracker::Params params_;
};