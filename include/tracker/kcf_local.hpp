#pragma once

#include <opencv2/core.hpp>

class KcfLocalTracker final {
public:
    struct Params {
        // kernel + training
        float sigma;
        float lambda;
        float interp_factor;
        float output_sigma_factor;

        // resize
        bool  resize;
        int   max_patch_size;

        // detection
        float detect_thresh;
        bool  wrap_kernel;

        // descriptors
        bool use_gray;   // NPCA
        bool use_cn;     // PCA path

        // PCA compression
        bool  compress_feature;
        int   compressed_size;     
        float pca_learning_rate;

        Params()
            : sigma(0.2f),
              lambda(1e-4f),
              interp_factor(0.075f),
              output_sigma_factor(1.0f / 16.0f),
              resize(true),
              max_patch_size(80 * 80),
              detect_thresh(0.5f),
              wrap_kernel(false),
              use_gray(true),
              use_cn(true),
              compress_feature(true),
              compressed_size(2),
              pca_learning_rate(0.15f) {}
    };

    KcfLocalTracker();
    explicit KcfLocalTracker(const Params& p);

    bool init(const cv::Mat& frame_bgr_or_gray, const cv::Rect& init_box);
    bool update(const cv::Mat& frame_bgr_or_gray, cv::Rect& out_box);

    bool getSearchWindowRect(cv::Rect& out) const;
    bool getLastResponse(cv::Mat& out_response32f) const;

    void reset();
    bool isInitialized() const { return initialized_; }

private:
    static void createHanningWindow(cv::Mat& dst, cv::Size winSize, int type);
    static void fft2(const cv::Mat& src, cv::Mat& dst);
    static void ifft2(const cv::Mat& src, cv::Mat& dst);

    bool getSubWindowGray(const cv::Mat& img, const cv::Rect& roi,
                          cv::Mat& feat_gray32f, cv::Mat& patch) const;

    bool getSubWindowCN(const cv::Mat& img_bgr, const cv::Rect& roi,
                        cv::Mat& feat_cn32fc10, cv::Mat& patch) const;
    static void extractCN(const cv::Mat& patch_bgr, cv::Mat& cn32fc10);

    static void updateProjectionMatrix(const cv::Mat& src_cn,
                                       cv::Mat& old_cov,
                                       cv::Mat& proj_matrix,
                                       float pca_rate,
                                       int compressed_sz);

    static void compressFeature(const cv::Mat& proj_matrix,
                                const cv::Mat& src,
                                cv::Mat& dst);

    void denseGaussKernel(float sigma, const cv::Mat& x, const cv::Mat& y, cv::Mat& k);
    void calcResponse(const cv::Mat& alphaf, const cv::Mat& kf, cv::Mat& response);

private:
    Params params_;
    bool initialized_ = false;
    bool resizeImage_ = false;
    int frame_ = 0;

    cv::Rect2d roi_;              // search window (padded), in possibly resized-image coords

    float output_sigma_ = 0.0f;

    cv::Mat hann_;                // CV_32F
    cv::Mat hann_cn_;             // CV_32FC10
    cv::Mat y_;                   // CV_32F
    cv::Mat yf_;                  // CV_32FC2

    cv::Mat x_;                   // merged features
    cv::Mat z_;                   // merged model features

    cv::Mat k_;                   // CV_32F
    cv::Mat kf_;                  // CV_32FC2
    cv::Mat alphaf_;              // CV_32FC2

    cv::Mat response_;            // CV_32F
    cv::Mat last_response_;       // CV_32F

    // PCA state (for CN)
    cv::Mat old_cov_mtx_;         // 10x10 CV_32F
    cv::Mat proj_mtx_;            // 10xK  CV_32F

    // Feature buffers
    cv::Mat X_gray_, Z_gray_;
    cv::Mat X_cn_,   Z_cn_;
    cv::Mat X_cn_c_, Z_cn_c_;

    // scratch
    cv::Mat imgPatch_;
    cv::Mat spec_;                // CV_32FC2
};