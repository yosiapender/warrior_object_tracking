#include <opencv2/opencv.hpp>

#include "detector/yolo_detector.hpp"
#include "tracker/kcf_tracker.hpp"
#include "tracker/csrt_tracker.hpp"

#include "app/config.hpp"
#include "app/tracking_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr int BOX_THICKNESS  = 2;
constexpr int TEXT_THICKNESS = 1;
constexpr int LINE_STYLE     = cv::LINE_AA;

inline bool pickBestDetection(const std::vector<Detection>& dets, Detection& best) {
    if (dets.empty()) return false;
    best = *std::max_element(
        dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) { return a.score < b.score; });
    return true;
}
} // namespace

int main(int argc, char** argv) {
    const app::AppConfig cfg = app::loadConfigFromArgsOrDefaults(argc, argv);
    app::printConfigBrief(cfg);

    const std::string onnx_path   = cfg.model_path;
    const std::string video_path  = cfg.video_path;
    const std::string output_path = cfg.out_video;
    const app::Policy  policy     = cfg.policy;

    std::cout << "=== YOLO + TRACKER VIDEO ===\n";
    std::cout << "[INFO] Video : " << video_path << "\n";
    std::cout << "[INFO] Out   : " << output_path << "\n";

    YoloDetector detector(onnx_path, 640, 640, policy.yolo_min_conf, 0.45f, /*use_cuda=*/true);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video: " << video_path << "\n";
        return -1;
    }

    double fps_in = cap.get(cv::CAP_PROP_FPS);
    if (fps_in <= 1.0) fps_in = 30.0;

    const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer(output_path,
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           fps_in,
                           cv::Size(w, h));
    if (!writer.isOpened()) {
        std::cerr << "[ERROR] Cannot open VideoWriter: " << output_path << "\n";
        return -1;
    }

    // Choose tracker here (kept simple; switch by editing one line)
    KcfTracker tracker;
    // CsrtTracker tracker;

    cv::Rect track_box{};
    bool have_track = false;

    app::Timer total_t, yolo_t, trk_t;

    int frame_idx = 0;
    cv::Rect prev_box{};
    bool have_prev = false;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        total_t.start();

        const bool periodic_redetect =
            (policy.redetect_every > 0 && (frame_idx % policy.redetect_every == 0));
        const bool need_yolo =
            !tracker.isInitialized() || !have_track || periodic_redetect;

        bool yolo_has = false;
        Detection best_det{};
        double yolo_ms = 0.0;
        double trk_ms  = 0.0;

        if (need_yolo) {
            yolo_t.start();
            const auto dets = detector.detect(frame);
            yolo_ms = yolo_t.stopMs();

            Detection best{};
            if (pickBestDetection(dets, best) && best.score >= policy.yolo_min_conf) {
                best_det = best;
                yolo_has = true;
            }
        }

        bool reinit = false;

        // ---- YOLO correction BEFORE tracker.update() ----
        if (yolo_has) {
            const cv::Rect yolo_box = app::clampRect(best_det.box, frame.size());
            const cv::Rect init_box = app::expandBox(yolo_box, policy.init_expand, frame.size());

            const bool must_init = !tracker.isInitialized() || !have_track;
            bool must_snap = false;

            if (!must_init && have_track) {
                const cv::Rect track_tight =
                    app::shrinkBox(track_box, 1.0f / policy.init_expand, frame.size());
                const double center_dist =
                    cv::norm(app::rectCenter(track_tight) - app::rectCenter(yolo_box));
                const double iou_val = app::IoU(track_tight, yolo_box);

                if (policy.snap_on_periodic && periodic_redetect) must_snap = true;
                if (center_dist > policy.snap_center_px) must_snap = true;
                if (iou_val < policy.snap_iou_min) must_snap = true;
            }

            if (must_init || must_snap) {
                tracker.reset();
                if (tracker.init(frame, init_box)) {
                    track_box  = init_box;
                    have_track = true;
                    reinit     = true;
                    prev_box   = track_box;
                    have_prev  = true;
                } else {
                    have_track = false;
                    have_prev  = false;
                }
            }
        }

        // ---- Tracker update ----
        if (!reinit && tracker.isInitialized() && have_track) {
            trk_t.start();
            cv::Rect new_box = track_box;
            bool ok = tracker.update(frame, new_box);
            trk_ms = trk_t.stopMs();

            if (ok && app::validBox(new_box, frame.size(), policy)) {
                if (have_prev) {
                    const double jump =
                        cv::norm(app::rectCenter(prev_box) - app::rectCenter(new_box));
                    if (jump > policy.max_center_jump_px) ok = false;
                }
            }

            if (ok && app::validBox(new_box, frame.size(), policy)) {
                track_box  = app::clampRect(new_box, frame.size());
                have_track = true;
                prev_box   = track_box;
                have_prev  = true;
            } else {
                have_track = false;
                have_prev  = false;
                tracker.reset();
            }
        }

        // ---- Draw ----
        if (yolo_has) {
            const cv::Rect yolo_box = app::clampRect(best_det.box, frame.size());
            cv::rectangle(frame, yolo_box, cv::Scalar(0, 255, 0), BOX_THICKNESS, LINE_STYLE);
            cv::putText(frame, "ball",
                        {yolo_box.x, std::max(0, yolo_box.y - 5)},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), TEXT_THICKNESS, LINE_STYLE);
        }

        if (have_track) {
            cv::rectangle(frame, track_box, cv::Scalar(255, 0, 0), BOX_THICKNESS, LINE_STYLE);
            cv::putText(frame, "ball",
                        {track_box.x, std::max(0, track_box.y - 5)},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 0, 0), TEXT_THICKNESS, LINE_STYLE);
        } else {
            cv::putText(frame, "TRACK LOST",
                        {10, 90}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 0, 255), 2, LINE_STYLE);
        }

        const double total_ms = total_t.stopMs();
        const double fps = (total_ms > 0.0) ? (1000.0 / total_ms) : 0.0;

        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)),
                    {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2, LINE_STYLE);

        cv::putText(frame, "Green=YOLO  Blue=TRK",
                    {10, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 255), 2, LINE_STYLE);

        app::drawTopRightStatus(frame,
                                std::string("YOLO: ") + (yolo_has ? "ON" : "OFF") +
                                    "  " + std::to_string(static_cast<int>(std::lround(yolo_ms))) + " ms",
                                0);
        app::drawTopRightStatus(frame,
                                std::string("TRK: ") + (have_track ? "ON" : "OFF") +
                                    "  " + std::to_string(static_cast<int>(std::lround(trk_ms))) + " ms",
                                1);

        cv::imshow("Ball tracking", frame);
        writer.write(frame);

        if (cv::waitKey(1) == 27) break;
        ++frame_idx;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "=== Finished ===\n";
    std::cout << "[INFO] Output saved to: " << output_path << "\n";
    return 0;
}
