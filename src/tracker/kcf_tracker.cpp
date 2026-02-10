#include "tracker/kcf_tracker.hpp"
#include <iostream>

KcfTracker::KcfTracker() {
    params_ = cv::TrackerKCF::Params();

    // Use CN in PCA pipeline so compressed_size <= channels is always true
    params_.desc_pca  = cv::TrackerKCF::CN;   // <-- change from GRAY to CN
    params_.desc_npca = cv::TrackerKCF::CN;

    // Keep compression valid (CN has many channels)
    params_.compress_feature = true;
    params_.compressed_size = 2;              // <= CN channels (safe)

    // Reasonable stability defaults
    params_.resize = true;
    params_.sigma = 0.2f;
    params_.lambda = 0.0001f;
    params_.interp_factor = 0.06f;            // a bit lower = less drift
    params_.output_sigma_factor = 0.1f;
    params_.pca_learning_rate = 0.15f;
}

bool KcfTracker::init(const cv::Mat& frame, const cv::Rect& init_box) {
    if (frame.empty()) {
        std::cerr << "[KCF(CN)] init failed: empty frame\n";
        initialized_ = false;
        return false;
    }
    if (init_box.width <= 1 || init_box.height <= 1) {
        std::cerr << "[KCF(CN)] init failed: invalid init_box\n";
        initialized_ = false;
        return false;
    }

    try {
        // Prefer params create; fallback if your OpenCV lacks it
        try {
            tracker_ = cv::TrackerKCF::create(params_);
        } catch (...) {
            tracker_ = cv::TrackerKCF::create();
        }

        // Your OpenCV expects cv::Rect (int) here
        cv::Rect box = init_box;
        tracker_->init(frame, box);

        initialized_ = true;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[KCF(CN)] init exception: " << e.what() << "\n";
        initialized_ = false;
        tracker_.release();
        return false;
    }
}

bool KcfTracker::update(const cv::Mat& frame, cv::Rect& out_box) {
    if (!initialized_ || tracker_.empty()) return false;
    if (frame.empty()) return false;

    bool ok = false;
    try {
        // MUST be cv::Rect lvalue because update wants cv::Rect&
        cv::Rect box = out_box;
        ok = tracker_->update(frame, box);
        out_box = box;
    } catch (const cv::Exception& e) {
        std::cerr << "[KCF(CN)] update exception: " << e.what() << "\n";
        ok = false;
    }

    if (!ok) {
        initialized_ = false;
        tracker_.release();
    }
    return ok;
}

void KcfTracker::reset() {
    initialized_ = false;
    tracker_.release();
}
