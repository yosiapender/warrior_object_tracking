#include <opencv2/opencv.hpp>
#include "detector/yolo_detector.hpp"
#include "tracker/kcf_tracker.hpp"
#include "tracker/csrt_tracker.hpp"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <string>

// --------------------
// Timer
// --------------------
class Timer {
public:
    void start() { t1 = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto t2 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
private:
    std::chrono::high_resolution_clock::time_point t1;
};

// --------------------
// Helpers
// --------------------
static bool pick_best_detection(const std::vector<Detection>& dets, Detection& best) {
    if (dets.empty()) return false;
    best = dets[0];
    for (const auto& d : dets)
        if (d.score > best.score) best = d;
    return true;
}

static cv::Rect clampRect(const cv::Rect& r, const cv::Size& sz) {
    return r & cv::Rect(0, 0, sz.width, sz.height);
}

static cv::Rect expandBox(const cv::Rect& b, float scale, const cv::Size& sz) {
    cv::Point2f c(b.x + b.width * 0.5f, b.y + b.height * 0.5f);
    int nw = std::max(2, (int)std::round(b.width  * scale));
    int nh = std::max(2, (int)std::round(b.height * scale));
    int x = (int)std::round(c.x - nw * 0.5f);
    int y = (int)std::round(c.y - nh * 0.5f);
    return clampRect(cv::Rect(x, y, nw, nh), sz);
}

static cv::Rect shrinkBox(const cv::Rect& b, float inv_scale, const cv::Size& sz) {
    return expandBox(b, inv_scale, sz);
}

static double IoU(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    double interA = (double)inter.area();
    double unionA = (double)a.area() + (double)b.area() - interA;
    return (unionA > 0.0) ? (interA / unionA) : 0.0;
}

static cv::Point2f rectCenter(const cv::Rect& r) {
    return cv::Point2f(r.x + r.width * 0.5f, r.y + r.height * 0.5f);
}

static void drawTopRightStatus(cv::Mat& frame,
                               const std::string& text,
                               int line_idx,
                               double scale = 0.8,
                               int thickness = 2,
                               int margin = 10,
                               int line_gap = 8) {
    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    int x = frame.cols - margin - ts.width;
    int y = margin + ts.height + line_idx * (ts.height + line_gap);
    cv::putText(frame, text, {x, y},
                cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
}

// --------------------
// Drawing style
// --------------------
constexpr int   BOX_THICKNESS  = 2;
constexpr int   TEXT_THICKNESS = 1;
constexpr int   LINE_STYLE     = cv::LINE_AA;

int main(int argc, char** argv) {
    // Usage:
    // ./object_tracking_video [video_path] [output_path]
    std::string onnx_path   = "../models/best.onnx";
    std::string video_path  = "../data/videos/video_30.avi";
    std::string output_path = "../data/output/no_obj.avi";

    if (argc >= 2) video_path  = argv[1];
    if (argc >= 3) output_path = argv[2];

    // --------------------
    // Policy knobs (video)
    // --------------------
    const int    REDETECT_EVERY   = 15;
    const float  INIT_EXPAND      = 1.25f;
    const float  YOLO_MIN_CONF    = 0.25f;
    const bool   SNAP_ON_PERIODIC = true;

    const double SNAP_CENTER_PX   = 30.0;
    const double SNAP_IOU_MIN     = 0.20;

    const double MAX_CENTER_JUMP_PX = 120.0;
    const double MIN_AREA_PX        = 16.0;
    const double MAX_AREA_FRAC      = 0.35;
    const double MIN_INFRAME_FRAC   = 0.65;

    std::cout << "=== YOLO + KCF VIDEO ===\n";
    std::cout << "[INFO] Video:  " << video_path << "\n";
    std::cout << "[INFO] Output: " << output_path << "\n";

    YoloDetector detector(onnx_path, 640, 640, 0.25f, 0.45f, true);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video: " << video_path << "\n";
        return -1;
    }

    double fps_in = cap.get(cv::CAP_PROP_FPS);
    if (fps_in <= 1.0) fps_in = 30.0;

    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc('M','J','P','G'),
        fps_in,
        cv::Size(w, h)
    );

    if (!writer.isOpened()) {
        std::cerr << "[ERROR] Cannot open VideoWriter: " << output_path << "\n";
        return -1;
    }

    auto valid_box = [&](const cv::Rect& r, const cv::Size& sz) -> bool {
        if (r.width <= 1 || r.height <= 1) return false;

        double area = (double)r.area();
        if (area < MIN_AREA_PX) return false;

        double frame_area = (double)sz.area();
        if (area > MAX_AREA_FRAC * frame_area) return false;

        cv::Rect cl = clampRect(r, sz);
        double in_frac = (area > 0.0) ? ((double)cl.area() / area) : 0.0;
        if (in_frac < MIN_INFRAME_FRAC) return false;

        return true;
    };

    KcfTracker tracker;
    //CsrtTracker tracker;

    cv::Rect track_box;
    bool have_track = false;

    Timer timer_total, timer_yolo, timer_trk;

    int frame_idx = 0;
    cv::Rect prev_box;
    bool have_prev = false;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        timer_total.start();

        bool periodic_redetect =
            (REDETECT_EVERY > 0 && frame_idx % REDETECT_EVERY == 0);

        bool need_yolo =
            !tracker.isInitialized() || !have_track || periodic_redetect;

        bool yolo_has = false;
        Detection best_det{};
        double yolo_ms = 0.0;
        double trk_ms  = 0.0;

        if (need_yolo) {
            timer_yolo.start();
            auto dets = detector.detect(frame);
            yolo_ms = timer_yolo.stop();

            Detection best;
            if (pick_best_detection(dets, best) && best.score >= YOLO_MIN_CONF) {
                best_det = best;
                yolo_has = true;
            }
        }

        bool reinit = false;

        // ---- YOLO correction BEFORE tracker.update() ----
        if (yolo_has) {
            cv::Rect yolo_box = clampRect(best_det.box, frame.size());
            cv::Rect init_box = expandBox(yolo_box, INIT_EXPAND, frame.size());

            bool must_init = !tracker.isInitialized() || !have_track;
            bool must_snap = false;

            if (!must_init && have_track) {
                cv::Rect track_tight = shrinkBox(track_box, 1.0f / INIT_EXPAND, frame.size());
                double center_dist = cv::norm(rectCenter(track_tight) - rectCenter(yolo_box));
                double iou_val = IoU(track_tight, yolo_box);

                if (SNAP_ON_PERIODIC && periodic_redetect) must_snap = true;
                if (center_dist > SNAP_CENTER_PX) must_snap = true;
                if (iou_val < SNAP_IOU_MIN) must_snap = true;
            }

            if (must_init || must_snap) {
                tracker.reset();
                if (tracker.init(frame, init_box)) {
                    track_box = init_box;
                    have_track = true;
                    reinit = true;
                    prev_box = track_box;
                    have_prev = true;
                } else {
                    have_track = false;
                    have_prev = false;
                }
            }
        }

        // ---- Tracker update ----
        if (!reinit && tracker.isInitialized() && have_track) {
            timer_trk.start();
            cv::Rect new_box = track_box;
            bool ok = tracker.update(frame, new_box);
            trk_ms = timer_trk.stop();

            if (ok && valid_box(new_box, frame.size())) {
                if (have_prev) {
                    double jump = cv::norm(rectCenter(prev_box) - rectCenter(new_box));
                    if (jump > MAX_CENTER_JUMP_PX) ok = false;
                }
            }

            if (ok && valid_box(new_box, frame.size())) {
                track_box = clampRect(new_box, frame.size());
                have_track = true;
                prev_box = track_box;
                have_prev = true;
            } else {
                have_track = false;
                have_prev = false;
                tracker.reset();
            }
        }

        // ---- Draw ----
        if (yolo_has) {
            cv::Rect yolo_box = clampRect(best_det.box, frame.size());
            cv::rectangle(frame, yolo_box, cv::Scalar(0,255,0), BOX_THICKNESS, LINE_STYLE);
            cv::putText(frame, "ball",
                        {yolo_box.x, std::max(0, yolo_box.y - 5)},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0,255,0), TEXT_THICKNESS, LINE_STYLE);
        }

        if (have_track) {
            cv::rectangle(frame, track_box, cv::Scalar(255,0,0), BOX_THICKNESS, LINE_STYLE);
            cv::putText(frame, "ball",
                        {track_box.x, std::max(0, track_box.y - 5)},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255,0,0), TEXT_THICKNESS, LINE_STYLE);
        } else {
            cv::putText(frame, "TRACK LOST",
                        {10, 90},
                        cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0,0,255), 2, LINE_STYLE);
        }

        double total_ms = timer_total.stop();
        double fps = (total_ms > 0.0) ? (1000.0 / total_ms) : 0.0;

        cv::putText(frame, "FPS: " + std::to_string((int)fps),
                    {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0,0,255), 2, LINE_STYLE);

        cv::putText(frame, "Green=YOLO  Blue=TRK",
                    {10, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0,0,255), 2, LINE_STYLE);

        drawTopRightStatus(frame, std::string("YOLO: ") + (yolo_has ? "ON" : "OFF") +
                                "  " + std::to_string((int)std::round(yolo_ms)) + " ms", 0);
        drawTopRightStatus(frame, std::string("TRK: ")  + (have_track ? "ON" : "OFF") +
                                "  " + std::to_string((int)std::round(trk_ms)) + " ms", 1);

        cv::imshow("Ball tracking", frame);
        writer.write(frame);

        if (cv::waitKey(1) == 27) break;
        frame_idx++;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "=== Finished ===\n";
    std::cout << "[INFO] Output saved to: " << output_path << "\n";
    return 0;
}
