#include "tracker/csrt_tracker.hpp"
#include <iostream>

CsrtTracker::CsrtTracker() = default;

bool CsrtTracker::init(const cv::Mat& frame, const cv::Rect& init_box) {
    if (frame.empty()) {
        std::cerr << "[CSRT] init failed: empty frame\n";
        initialized_ = false;
        return false;
    }
    if (init_box.width <= 1 || init_box.height <= 1) {
        std::cerr << "[CSRT] init failed: invalid init_box\n";
        initialized_ = false;
        return false;
    }

    try {
        tracker_ = cv::TrackerCSRT::create();

        // Your OpenCV build expects cv::Rect& for init/update via cv::Tracker base
        cv::Rect box = init_box;
        tracker_->init(frame, box);

        initialized_ = true;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[CSRT] init exception: " << e.what() << "\n";
        initialized_ = false;
        tracker_.release();
        return false;
    }
}

bool CsrtTracker::update(const cv::Mat& frame, cv::Rect& out_box) {
    if (!initialized_ || tracker_.empty()) return false;
    if (frame.empty()) return false;

    bool ok = false;
    try {
        cv::Rect box = out_box;              // must be lvalue
        ok = tracker_->update(frame, box);   // expects cv::Rect&
        out_box = box;
    } catch (const cv::Exception& e) {
        std::cerr << "[CSRT] update exception: " << e.what() << "\n";
        ok = false;
    }

    if (!ok) {
        initialized_ = false;
        tracker_.release();
    }
    return ok;
}

void CsrtTracker::reset() {
    initialized_ = false;
    tracker_.release();
}
