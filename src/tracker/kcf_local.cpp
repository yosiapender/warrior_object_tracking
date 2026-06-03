#include "tracker/kcf_local.hpp"
#include "tracker/colornames_lut.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

KcfLocalTracker::KcfLocalTracker() : params_(Params()) {}
KcfLocalTracker::KcfLocalTracker(const Params& p) : params_(p) {}

void KcfLocalTracker::reset() {
    initialized_ = false;
    resizeImage_ = false;
    frame_ = 0;
    roi_ = cv::Rect2d();

    output_sigma_ = 0.0f;

    hann_.release();
    hann_cn_.release();
    y_.release();
    yf_.release();

    x_.release();
    z_.release();
    k_.release();
    kf_.release();
    alphaf_.release();

    response_.release();
    last_response_.release();

    old_cov_mtx_.release();
    proj_mtx_.release();

    X_gray_.release(); Z_gray_.release();
    X_cn_.release();   Z_cn_.release();
    X_cn_c_.release(); Z_cn_c_.release();

    imgPatch_.release();
    spec_.release();
}

bool KcfLocalTracker::init(const cv::Mat& frame, const cv::Rect& init_box) {
    reset();

    if (frame.empty()) return false;
    if (init_box.width <= 1 || init_box.height <= 1) return false;

    if (params_.use_cn && frame.channels() != 3) {
        return false;
    }

    frame_ = 0;

    roi_.x = static_cast<double>(init_box.x);
    roi_.y = static_cast<double>(init_box.y);
    roi_.width  = static_cast<double>(init_box.width);
    roi_.height = static_cast<double>(init_box.height);

    output_sigma_ = std::sqrt(static_cast<float>(roi_.width * roi_.height)) * params_.output_sigma_factor;
    output_sigma_ = -0.5f / (output_sigma_ * output_sigma_);

    resizeImage_ = false;
    if (params_.resize && (roi_.width * roi_.height > params_.max_patch_size)) {
        resizeImage_ = true;
        roi_.x      /= 2.0;
        roi_.y      /= 2.0;
        roi_.width  /= 2.0;
        roi_.height /= 2.0;
    }

    roi_.x -= roi_.width / 2.0;
    roi_.y -= roi_.height / 2.0;
    roi_.width  *= 2.0;
    roi_.height *= 2.0;

    const int imgW = resizeImage_ ? frame.cols / 2 : frame.cols;
    const int imgH = resizeImage_ ? frame.rows / 2 : frame.rows;
    if ((roi_ & cv::Rect2d(0, 0, imgW, imgH)).empty()) return false;

    createHanningWindow(hann_, roi_.size(), CV_32F);
    if (params_.use_cn) {
        cv::Mat layers[10];
        for (int i = 0; i < 10; ++i) layers[i] = hann_;
        cv::merge(layers, 10, hann_cn_);
    }

    y_ = cv::Mat::zeros(static_cast<int>(roi_.height), static_cast<int>(roi_.width), CV_32F);
    const int cy = y_.rows / 2;
    const int cx = y_.cols / 2;
    for (int i = 0; i < y_.rows; ++i) {
        float* row = y_.ptr<float>(i);
        const int dy = i - cy + 1;
        for (int j = 0; j < y_.cols; ++j) {
            const int dx = j - cx + 1;
            row[j] = static_cast<float>(dy * dy + dx * dx);
        }
    }
    y_ *= output_sigma_;
    cv::exp(y_, y_);
    fft2(y_, yf_);

    initialized_ = true;
    last_response_ = cv::Mat::zeros(y_.size(), CV_32F);
    return true;
}

bool KcfLocalTracker::getSearchWindowRect(cv::Rect& out) const {
    if (!initialized_) return false;

    cv::Rect2d r = roi_;
    if (resizeImage_) {
        r.x *= 2.0; r.y *= 2.0; r.width *= 2.0; r.height *= 2.0;
    }
    out = cv::Rect(
        static_cast<int>(std::round(r.x)),
        static_cast<int>(std::round(r.y)),
        static_cast<int>(std::round(r.width)),
        static_cast<int>(std::round(r.height))
    );
    return true;
}

bool KcfLocalTracker::getLastResponse(cv::Mat& out) const {
    if (last_response_.empty()) return false;
    out = last_response_;
    return true;
}

bool KcfLocalTracker::update(const cv::Mat& frame, cv::Rect& out_box) {
    if (!initialized_ || frame.empty()) return false;

    if (params_.use_cn && frame.channels() != 3) return false;

    cv::Mat img;

    if (resizeImage_) {
        cv::resize(frame, img, cv::Size(frame.cols / 2, frame.rows / 2), 0, 0, cv::INTER_LINEAR_EXACT);
    } else {
        img = frame;
    }

    if (frame_ > 0) {
        if (params_.use_gray) {
            if (!getSubWindowGray(img, roi_, X_gray_, imgPatch_)) return false;
        }
        if (params_.use_cn) {
            if (!getSubWindowCN(img, roi_, X_cn_, imgPatch_)) return false;

            if (params_.compress_feature) {
                compressFeature(proj_mtx_, X_cn_, X_cn_c_);
                compressFeature(proj_mtx_, Z_cn_, Z_cn_c_);
            } else {
                X_cn_c_ = X_cn_;
                Z_cn_c_ = Z_cn_;
            }
        }

        if (params_.use_gray && params_.use_cn) {
            cv::Mat partsX[2] = { X_cn_c_, X_gray_ };
            cv::merge(partsX, 2, x_);
            cv::Mat partsZ[2] = { Z_cn_c_, Z_gray_ };
            cv::merge(partsZ, 2, z_);
        } else if (params_.use_cn) {
            x_ = X_cn_c_;
            z_ = Z_cn_c_;
        } else {
            x_ = X_gray_;
            z_ = Z_gray_;
        }

        denseGaussKernel(params_.sigma, x_, z_, k_);
        fft2(k_, kf_);
        calcResponse(alphaf_, kf_, response_);
        response_.copyTo(last_response_);

        double minV=0.0, maxV=0.0;
        cv::Point minL, maxL;
        cv::minMaxLoc(response_, &minV, &maxV, &minL, &maxL);

        if (static_cast<float>(maxV) < params_.detect_thresh) {
            return false;
        }

        roi_.x += (maxL.x - roi_.width / 2.0 + 1.0);
        roi_.y += (maxL.y - roi_.height / 2.0 + 1.0);
    }

    cv::Rect2d bbox;
    bbox.x = (resizeImage_ ? roi_.x * 2.0 : roi_.x) + (resizeImage_ ? roi_.width * 2.0 : roi_.width) / 4.0;
    bbox.y = (resizeImage_ ? roi_.y * 2.0 : roi_.y) + (resizeImage_ ? roi_.height * 2.0 : roi_.height) / 4.0;
    bbox.width  = (resizeImage_ ? roi_.width * 2.0 : roi_.width) / 2.0;
    bbox.height = (resizeImage_ ? roi_.height * 2.0 : roi_.height) / 2.0;

    if (params_.use_gray) {
        if (!getSubWindowGray(img, roi_, X_gray_, imgPatch_)) return false;
        if (frame_ == 0) Z_gray_ = X_gray_.clone();
        else Z_gray_ = (1.0f - params_.interp_factor) * Z_gray_ + params_.interp_factor * X_gray_;
    }

    if (params_.use_cn) {
        if (!getSubWindowCN(img, roi_, X_cn_, imgPatch_)) return false;

        if (frame_ == 0) Z_cn_ = X_cn_.clone();
        else Z_cn_ = (1.0f - params_.interp_factor) * Z_cn_ + params_.interp_factor * X_cn_;

        if (params_.compress_feature) {
            updateProjectionMatrix(Z_cn_, old_cov_mtx_, proj_mtx_,
                                   params_.pca_learning_rate, params_.compressed_size);
            compressFeature(proj_mtx_, X_cn_, X_cn_c_);
        } else {
            X_cn_c_ = X_cn_;
        }
    }

    if (params_.use_gray && params_.use_cn) {
        cv::Mat partsX[2] = { X_cn_c_, X_gray_ };
        cv::merge(partsX, 2, x_);
    } else if (params_.use_cn) {
        x_ = X_cn_c_;
    } else {
        x_ = X_gray_;
    }

    denseGaussKernel(params_.sigma, x_, x_, k_);
    fft2(k_, kf_);

    cv::Mat kf_lambda = kf_.clone();
    for (int i = 0; i < kf_lambda.rows; ++i) {
        cv::Vec2f* row = kf_lambda.ptr<cv::Vec2f>(i);
        for (int j = 0; j < kf_lambda.cols; ++j) row[j][0] += params_.lambda;
    }

    cv::Mat new_alphaf(yf_.rows, yf_.cols, CV_32FC2);
    for (int i = 0; i < yf_.rows; ++i) {
        const cv::Vec2f* yrow = yf_.ptr<cv::Vec2f>(i);
        const cv::Vec2f* krow = kf_lambda.ptr<cv::Vec2f>(i);
        cv::Vec2f* arow = new_alphaf.ptr<cv::Vec2f>(i);

        for (int j = 0; j < yf_.cols; ++j) {
            const float a = yrow[j][0], b = yrow[j][1];
            const float c = krow[j][0], d = krow[j][1];
            const float denom = 1.0f / (c * c + d * d + 1e-12f);

            arow[j][0] = (a * c + b * d) * denom;
            arow[j][1] = (b * c - a * d) * denom;
        }
    }

    if (frame_ == 0) {
        alphaf_ = new_alphaf.clone();
    } else {
        alphaf_ = (1.0f - params_.interp_factor) * alphaf_ + params_.interp_factor * new_alphaf;
    }

    if (params_.use_cn) {
        if (params_.compress_feature) {
            compressFeature(proj_mtx_, Z_cn_, Z_cn_c_);
        } else {
            Z_cn_c_ = Z_cn_;
        }
    }

    frame_++;

    const cv::Rect frame_rect(0, 0, frame.cols, frame.rows);
    out_box = cv::Rect(
        static_cast<int>(std::round(bbox.x)),
        static_cast<int>(std::round(bbox.y)),
        static_cast<int>(std::round(bbox.width)),
        static_cast<int>(std::round(bbox.height))
    ) & frame_rect;

    return true;
}

void KcfLocalTracker::createHanningWindow(cv::Mat& dst, cv::Size winSize, int type) {
    CV_Assert(type == CV_32FC1 || type == CV_64FC1);
    dst.create(winSize, type);

    const int rows = dst.rows;
    const int cols = dst.cols;

    const float coeff0 = 2.0f * static_cast<float>(CV_PI) / (cols - 1);
    const float coeff1 = 2.0f * static_cast<float>(CV_PI) / (rows - 1);

    std::vector<float> wc(cols);
    for (int j = 0; j < cols; ++j) wc[j] = 0.5f * (1.0f - std::cos(coeff0 * j));

    for (int i = 0; i < rows; ++i) {
        float* row = dst.ptr<float>(i);
        const float wr = 0.5f * (1.0f - std::cos(coeff1 * i));
        for (int j = 0; j < cols; ++j) row[j] = wr * wc[j];
    }
}

void KcfLocalTracker::fft2(const cv::Mat& src, cv::Mat& dst) {
    cv::dft(src, dst, cv::DFT_COMPLEX_OUTPUT);
}

void KcfLocalTracker::ifft2(const cv::Mat& src, cv::Mat& dst) {
    cv::idft(src, dst, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
}

bool KcfLocalTracker::getSubWindowGray(const cv::Mat& img, const cv::Rect& roi_in,
                                      cv::Mat& feat32f, cv::Mat& patch) const {
    cv::Rect region = roi_in;
    if ((roi_in & cv::Rect(0, 0, img.cols, img.rows)).empty()) return false;

    if (roi_in.x < 0) { region.x = 0; region.width += roi_in.x; }
    if (roi_in.y < 0) { region.y = 0; region.height += roi_in.y; }
    if (roi_in.x + roi_in.width > img.cols) region.width = img.cols - roi_in.x;
    if (roi_in.y + roi_in.height > img.rows) region.height = img.rows - roi_in.y;

    if (region.empty()) return false;
    patch = img(region).clone();

    const int addTop    = region.y - roi_in.y;
    const int addLeft   = region.x - roi_in.x;
    const int addBottom = (roi_in.y + roi_in.height > img.rows) ? (roi_in.y + roi_in.height - img.rows) : 0;
    const int addRight  = (roi_in.x + roi_in.width  > img.cols) ? (roi_in.x + roi_in.width  - img.cols) : 0;

    cv::copyMakeBorder(patch, patch, addTop, addBottom, addLeft, addRight, cv::BORDER_REPLICATE);
    if (patch.empty()) return false;

    cv::Mat gray;
    if (patch.channels() == 3) cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    else gray = patch;

    gray.convertTo(feat32f, CV_32F, 1.0 / 255.0, -0.5);
    feat32f = feat32f.mul(hann_);
    return true;
}

void KcfLocalTracker::extractCN(const cv::Mat& patch_bgr, cv::Mat& cn32fc10) {
    CV_Assert(patch_bgr.type() == CV_8UC3);
    cn32fc10.create(patch_bgr.rows, patch_bgr.cols, CV_32FC(10));

    for (int y = 0; y < patch_bgr.rows; ++y) {
        const cv::Vec3b* row = patch_bgr.ptr<cv::Vec3b>(y);
        cv::Vec<float,10>* out = cn32fc10.ptr<cv::Vec<float,10>>(y);
        for (int x = 0; x < patch_bgr.cols; ++x) {
            const cv::Vec3b p = row[x];
            const int r = static_cast<int>(p[2] >> 3);
            const int g = static_cast<int>(p[1] >> 3);
            const int b = static_cast<int>(p[0] >> 3);
            const int idx = r + 32 * g + 32 * 32 * b;

            const float* lut = kcf_cn::ColorNames[idx];
            for (int k = 0; k < 10; ++k) out[x][k] = lut[k];
        }
    }
}

bool KcfLocalTracker::getSubWindowCN(const cv::Mat& img, const cv::Rect& roi_in,
                                    cv::Mat& feat_cn32fc10, cv::Mat& patch) const {
    CV_Assert(img.channels() == 3);

    cv::Rect region = roi_in;
    if ((roi_in & cv::Rect(0, 0, img.cols, img.rows)).empty()) return false;

    if (roi_in.x < 0) { region.x = 0; region.width += roi_in.x; }
    if (roi_in.y < 0) { region.y = 0; region.height += roi_in.y; }
    if (roi_in.x + roi_in.width > img.cols) region.width = img.cols - roi_in.x;
    if (roi_in.y + roi_in.height > img.rows) region.height = img.rows - roi_in.y;

    if (region.empty()) return false;
    patch = img(region).clone();

    const int addTop    = region.y - roi_in.y;
    const int addLeft   = region.x - roi_in.x;
    const int addBottom = (roi_in.y + roi_in.height > img.rows) ? (roi_in.y + roi_in.height - img.rows) : 0;
    const int addRight  = (roi_in.x + roi_in.width  > img.cols) ? (roi_in.x + roi_in.width  - img.cols) : 0;

    cv::copyMakeBorder(patch, patch, addTop, addBottom, addLeft, addRight, cv::BORDER_REPLICATE);
    if (patch.empty()) return false;

    extractCN(patch, feat_cn32fc10);
    feat_cn32fc10 = feat_cn32fc10.mul(hann_cn_);
    return true;
}

void KcfLocalTracker::updateProjectionMatrix(const cv::Mat& src_cn,
                                             cv::Mat& old_cov,
                                             cv::Mat& proj_matrix,
                                             float pca_rate,
                                             int compressed_sz) {
    CV_Assert(src_cn.depth() == CV_32F);
    const int C = src_cn.channels();
    CV_Assert(compressed_sz <= C);

    std::vector<cv::Mat> ch;
    cv::split(src_cn, ch);

    for (int i = 0; i < C; ++i) {
        const cv::Scalar m = cv::mean(ch[i]);
        ch[i] -= m;
    }

    cv::Mat data;
    cv::merge(ch, data);
    data = data.reshape(1, data.rows * data.cols); // (N x C)

    const int N = std::max(1, data.rows - 1);
    cv::Mat new_cov = (1.0f / static_cast<float>(N)) * (data.t() * data);

    if (old_cov.empty()) old_cov = new_cov.clone();
    cv::Mat cov = (1.0f - pca_rate) * old_cov + pca_rate * new_cov;

    cv::Mat w, u, vt;
    cv::SVD::compute(cov, w, u, vt, cv::SVD::MODIFY_A);

    proj_matrix = u(cv::Rect(0, 0, compressed_sz, C)).clone(); // CxK
    old_cov = cov;
}

void KcfLocalTracker::compressFeature(const cv::Mat& proj_matrix,
                                      const cv::Mat& src,
                                      cv::Mat& dst) {
    CV_Assert(src.depth() == CV_32F);
    CV_Assert(proj_matrix.depth() == CV_32F);

    cv::Mat data = src.reshape(1, src.rows * src.cols); // (N x C)
    cv::Mat compressed = data * proj_matrix;            // (N x K)
    dst = compressed.reshape(proj_matrix.cols, src.rows).clone(); // HxWxK
}

void KcfLocalTracker::denseGaussKernel(float sigma, const cv::Mat& x, const cv::Mat& y, cv::Mat& k) {
    CV_Assert(x.depth() == CV_32F && y.depth() == CV_32F);
    CV_Assert(x.size() == y.size() && x.type() == y.type());

    const int C = x.channels();
    CV_Assert(C >= 1);

    std::vector<cv::Mat> xch, ych;
    cv::split(x, xch);
    cv::split(y, ych);

    std::vector<cv::Mat> xyf(C);
    cv::Mat sum_xyf;

    for (int c = 0; c < C; ++c) {
        cv::Mat xf, yf;
        cv::dft(xch[c], xf, cv::DFT_COMPLEX_OUTPUT);
        cv::dft(ych[c], yf, cv::DFT_COMPLEX_OUTPUT);

        cv::mulSpectrums(xf, yf, xyf[c], 0, /*conjB=*/true);

        if (c == 0) sum_xyf = xyf[c].clone();
        else sum_xyf += xyf[c];
    }

    cv::Mat xy;
    cv::idft(sum_xyf, xy, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    if (params_.wrap_kernel) {
        const int shift_y = xy.rows / 2;
        const int shift_x = xy.cols / 2;

        // shift rows
        if (shift_y > 0) {
            cv::Mat tmp;
            xy(cv::Rect(0, 0, xy.cols, shift_y)).copyTo(tmp);
            xy(cv::Rect(0, shift_y, xy.cols, xy.rows - shift_y)).copyTo(xy(cv::Rect(0, 0, xy.cols, xy.rows - shift_y)));
            tmp.copyTo(xy(cv::Rect(0, xy.rows - shift_y, xy.cols, shift_y)));
        }
        // shift cols
        if (shift_x > 0) {
            cv::Mat tmp;
            xy(cv::Rect(0, 0, shift_x, xy.rows)).copyTo(tmp);
            xy(cv::Rect(shift_x, 0, xy.cols - shift_x, xy.rows)).copyTo(xy(cv::Rect(0, 0, xy.cols - shift_x, xy.rows)));
            tmp.copyTo(xy(cv::Rect(xy.cols - shift_x, 0, shift_x, xy.rows)));
        }
    }

    const double nx = cv::norm(x);
    const double ny = cv::norm(y);
    const double xx = nx * nx;
    const double yy = ny * ny;

    const float invN = 1.0f / static_cast<float>(x.rows * x.cols * C);
    xy = (static_cast<float>(xx + yy) - 2.0f * xy) * invN;

    for (int i = 0; i < xy.rows; ++i) {
        float* row = xy.ptr<float>(i);
        for (int j = 0; j < xy.cols; ++j) row[j] = std::max(0.0f, row[j]);
    }

    xy *= (-1.0f / (sigma * sigma));
    cv::exp(xy, k);
}

void KcfLocalTracker::calcResponse(const cv::Mat& alphaf, const cv::Mat& kf, cv::Mat& response) {
    cv::mulSpectrums(alphaf, kf, spec_, 0, false);
    ifft2(spec_, response);
}