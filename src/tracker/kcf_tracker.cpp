#include "tracker/kcf_tracker.hpp"
#include <iostream>

KcfTracker::KcfTracker() {
    params_ = KcfLocalTracker::Params();

    // core KCF params
    params_.sigma = 0.2f;
    params_.lambda = 1e-4f;
    params_.interp_factor = 0.06f;
    params_.output_sigma_factor = 0.1f;
    params_.detect_thresh = 0.5f;

    // CN/PCA toggles
    params_.use_gray = true;
    params_.use_cn = true;                 // <-- enable CN
    params_.compress_feature = true;       // <-- enable PCA compression
    params_.compressed_size = 2;           // typical
    params_.pca_learning_rate = 0.15f;     // typical

    params_.wrap_kernel = false;
    params_.resize = true;
    params_.max_patch_size = 80 * 80;
}

bool KcfTracker::init(const cv::Mat& frame, const cv::Rect& init_box) {
    if (frame.empty() || init_box.width <= 1 || init_box.height <= 1) {
        initialized_ = false;
        tracker_.reset();
        return false;
    }

    try {
        tracker_ = std::make_unique<KcfLocalTracker>(params_);
        if (!tracker_->init(frame, init_box)) {
            initialized_ = false;
            tracker_.reset();
            return false;
        }
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[KCF(Local)] init exception: " << e.what() << "\n";
        initialized_ = false;
        tracker_.reset();
        return false;
    }
}

bool KcfTracker::update(const cv::Mat& frame, cv::Rect& out_box) {
    if (!initialized_ || !tracker_ || frame.empty()) return false;

    bool ok = false;
    try {
        ok = tracker_->update(frame, out_box);
    } catch (const std::exception& e) {
        std::cerr << "[KCF(Local)] update exception: " << e.what() << "\n";
        ok = false;
    }

    if (!ok) {
        initialized_ = false;
        tracker_.reset();
    }
    return ok;
}

bool KcfTracker::getSearchWindowRect(cv::Rect& out) const {
    if (!tracker_) return false;
    return tracker_->getSearchWindowRect(out);
}

bool KcfTracker::getLastResponseMap(cv::Mat& out_response32f) const {
    if (!tracker_) return false;
    return tracker_->getLastResponse(out_response32f);
}

void KcfTracker::reset() {
    initialized_ = false;
    tracker_.reset();
}