#include <opencv2/opencv.hpp>
#include "detector/yolo_detector.hpp"
#include "tracker/kcf_tracker.hpp"
#include "tracker/csrt_tracker.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <memory>

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
// Helpers (same as your main)
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
// CVAT XML parser (video 1.1) for one label
// --------------------
static bool getAttr(const std::string& s, const std::string& key, std::string& out) {
    std::string pat = key + "=\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    size_t q = s.find('"', p);
    if (q == std::string::npos) return false;
    out = s.substr(p, q - p);
    return true;
}

struct GtFrame {
    bool has = false;
    cv::Rect box;
};

static std::unordered_map<int, GtFrame> loadCvatXmlTrackBoxes(
    const std::string& xml_path,
    const std::string& target_label
) {
    std::ifstream fin(xml_path);
    if (!fin.is_open()) throw std::runtime_error("Cannot open XML: " + xml_path);

    std::unordered_map<int, GtFrame> gt;

    bool in_track = false;
    std::string current_label;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.find("<track") != std::string::npos) {
            in_track = true;
            current_label.clear();
            std::string lbl;
            if (getAttr(line, "label", lbl)) current_label = lbl;
            continue;
        }
        if (line.find("</track>") != std::string::npos) {
            in_track = false;
            current_label.clear();
            continue;
        }

        if (!in_track) continue;
        if (current_label != target_label) continue;

        if (line.find("<box") != std::string::npos) {
            std::string s_frame, s_xtl, s_ytl, s_xbr, s_ybr, s_outside;
            if (!getAttr(line, "frame", s_frame)) continue;
            (void)getAttr(line, "outside", s_outside);
            if (!getAttr(line, "xtl", s_xtl)) continue;
            if (!getAttr(line, "ytl", s_ytl)) continue;
            if (!getAttr(line, "xbr", s_xbr)) continue;
            if (!getAttr(line, "ybr", s_ybr)) continue;

            int frame_idx = std::stoi(s_frame);
            int outside = (!s_outside.empty()) ? std::stoi(s_outside) : 0;

            if (outside == 1) {
                gt[frame_idx] = GtFrame{false, cv::Rect()};
                continue;
            }

            double xtl = std::stod(s_xtl);
            double ytl = std::stod(s_ytl);
            double xbr = std::stod(s_xbr);
            double ybr = std::stod(s_ybr);

            int x = (int)std::round(xtl);
            int y = (int)std::round(ytl);
            int w = std::max(1, (int)std::round(xbr - xtl));
            int h = std::max(1, (int)std::round(ybr - ytl));

            gt[frame_idx] = GtFrame{true, cv::Rect(x, y, w, h)};
        }
    }
    return gt;
}

// --------------------
// Tracker runtime switch (kcf / csrt)
// --------------------
struct ITracker {
    virtual ~ITracker() = default;
    virtual bool init(const cv::Mat& frame, const cv::Rect& box) = 0;
    virtual bool update(const cv::Mat& frame, cv::Rect& box) = 0;
    virtual void reset() = 0;
    virtual bool isInitialized() const = 0;
};

struct KcfAdapter : ITracker {
    KcfTracker t;
    bool init(const cv::Mat& f, const cv::Rect& b) override { return t.init(f, b); }
    bool update(const cv::Mat& f, cv::Rect& b) override { return t.update(f, b); }
    void reset() override { t.reset(); }
    bool isInitialized() const override { return t.isInitialized(); }
};

struct CsrtAdapter : ITracker {
    CsrtTracker t;
    bool init(const cv::Mat& f, const cv::Rect& b) override { return t.init(f, b); }
    bool update(const cv::Mat& f, cv::Rect& b) override { return t.update(f, b); }
    void reset() override { t.reset(); }
    bool isInitialized() const override { return t.isInitialized(); }
};

// --------------------
// Drawing style
// --------------------
constexpr int BOX_THICKNESS  = 2;
constexpr int TEXT_THICKNESS = 1;
constexpr int LINE_STYLE     = cv::LINE_AA;

// --------------------
// MAIN
//
// Usage:
// ./object_tracking_eval_cvat_hybrid <video> <xml> <label> <kcf|csrt> <csv_out> [vis_out]
//
// Example:
// ./object_tracking_eval_cvat_hybrid ../data/videos/1_occlusion.avi ../data/gt/video1.1.xml ball csrt ../data/output/eval.csv ../data/output/eval_vis.avi
// --------------------
int main(int argc, char** argv) {
    std::string onnx_path   = "../models/model_v3.onnx";
    std::string video_path  = (argc >= 2) ? argv[1] : "../data/videos/1_occlusion.avi";
    std::string xml_path    = (argc >= 3) ? argv[2] : "../data/gt/video1.1.xml";
    std::string label       = (argc >= 4) ? argv[3] : "ball";
    std::string tracker_ty  = (argc >= 5) ? argv[4] : "csrt";
    std::string csv_out     = (argc >= 6) ? argv[5] : "../data/output/eval_hybrid.csv";
    std::string vis_out     = (argc >= 7) ? argv[6] : ""; // optional annotated video

    // --------------------
    // Policy knobs (same as your main)
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

    // Evaluation thresholds
    const double IOU_SUCCESS_THR = 0.50;
    const double CENTER_OK_PX    = 20.0;

    std::cout << "=== YOLO + TRACKER (HYBRID) + CVAT EVAL ===\n";
    std::cout << "[INFO] Video   : " << video_path << "\n";
    std::cout << "[INFO] XML     : " << xml_path << "\n";
    std::cout << "[INFO] Label   : " << label << "\n";
    std::cout << "[INFO] Tracker : " << tracker_ty << "\n";
    std::cout << "[INFO] CSV out : " << csv_out << "\n";
    if (!vis_out.empty()) std::cout << "[INFO] Vis out : " << vis_out << "\n";

    // Load GT
    std::unordered_map<int, GtFrame> gt;
    try {
        gt = loadCvatXmlTrackBoxes(xml_path, label);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return -1;
    }

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

    // Optional visualization video writer
    cv::VideoWriter vis_writer;
    if (!vis_out.empty()) {
        vis_writer.open(vis_out,
                        cv::VideoWriter::fourcc('M','J','P','G'),
                        fps_in,
                        cv::Size(w, h));
        if (!vis_writer.isOpened()) {
            std::cerr << "[WARN] Cannot open VideoWriter: " << vis_out << "\n";
        }
    }

    // CSV
    std::ofstream fout(csv_out);
    if (!fout.is_open()) {
        std::cerr << "[ERROR] Cannot write CSV: " << csv_out << "\n";
        return -1;
    }
    fout << "frame,gt_has,pred_has,iou,center_dist_px,yolo_has,reinit\n";

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

    // Tracker switch
    std::unique_ptr<ITracker> tracker;
    if (tracker_ty == "kcf") tracker = std::make_unique<KcfAdapter>();
    else if (tracker_ty == "csrt") tracker = std::make_unique<CsrtAdapter>();
    else {
        std::cerr << "[ERROR] Unknown tracker type: " << tracker_ty << " (use kcf|csrt)\n";
        return -1;
    }

    cv::Rect track_box;
    bool have_track = false;

    Timer timer_total, timer_yolo, timer_trk;

    int frame_idx = 0;
    cv::Rect prev_box;
    bool have_prev = false;

    // Stats
    int total_frames = 0;
    int gt_present_frames = 0;
    int gt_absent_frames = 0;
    int pred_present_frames = 0;

    int matched_frames = 0;
    int success_iou_frames = 0;
    double sum_iou = 0.0;

    int center_ok_frames = 0;
    double sum_center = 0.0;

    int fp_on_absent = 0;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        total_frames++;
        timer_total.start();

        // ---- GT ----
        bool gt_has = false;
        cv::Rect gt_box;
        auto it = gt.find(frame_idx);
        if (it != gt.end() && it->second.has) {
            gt_has = true;
            gt_box = clampRect(it->second.box, frame.size());
        }
        if (gt_has) gt_present_frames++;
        else gt_absent_frames++;

        bool periodic_redetect = (REDETECT_EVERY > 0 && frame_idx % REDETECT_EVERY == 0);
        bool need_yolo = !tracker->isInitialized() || !have_track || periodic_redetect;

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

            bool must_init = !tracker->isInitialized() || !have_track;
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
                tracker->reset();
                if (tracker->init(frame, init_box)) {
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
        if (!reinit && tracker->isInitialized() && have_track) {
            timer_trk.start();
            cv::Rect new_box = track_box;
            bool ok = tracker->update(frame, new_box);
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
                tracker->reset();
            }
        }

        bool pred_has = have_track && tracker->isInitialized();
        if (pred_has) pred_present_frames++;

        // ---- Eval metrics ----
        double iou = -1.0;
        double cd  = -1.0;

        if (gt_has && pred_has) {
            matched_frames++;
            iou = IoU(gt_box, track_box);
            cd  = cv::norm(rectCenter(gt_box) - rectCenter(track_box));
            sum_iou += iou;
            sum_center += cd;
            if (iou >= IOU_SUCCESS_THR) success_iou_frames++;
            if (cd <= CENTER_OK_PX) center_ok_frames++;
        } else if (!gt_has && pred_has) {
            fp_on_absent++;
        }

        fout << frame_idx << ","
             << (gt_has ? 1 : 0) << ","
             << (pred_has ? 1 : 0) << ","
             << iou << ","
             << cd << ","
             << (yolo_has ? 1 : 0) << ","
             << (reinit ? 1 : 0) << "\n";

        // ---- Visualization ----
        cv::Mat vis = frame.clone();

        if (gt_has) {
            cv::rectangle(vis, gt_box, cv::Scalar(0,255,255), BOX_THICKNESS, LINE_STYLE); // yellow
        }
        
        if (pred_has) {
            cv::rectangle(vis, track_box, cv::Scalar(255,0,0), BOX_THICKNESS, LINE_STYLE);
        } else {
            cv::putText(vis, "TRACK LOST",
                        {10, 90},
                        cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0,0,255), 2, LINE_STYLE);
        }

        double total_ms = timer_total.stop();
        double fps = (total_ms > 0.0) ? (1000.0 / total_ms) : 0.0;

        cv::putText(vis, "FPS: " + std::to_string((int)fps),
                    {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0,0,255), 2, LINE_STYLE);

        cv::putText(vis, "Yellow=GT  Blue=YOLO+TRK",
                    {10, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0,0,255), 2, LINE_STYLE);

        drawTopRightStatus(vis, std::string("YOLO: ") + (yolo_has ? "ON" : "OFF") +
                                   "  " + std::to_string((int)std::round(yolo_ms)) + " ms", 0);
        drawTopRightStatus(vis, std::string("TRK: ")  + (pred_has ? "ON" : "OFF") +
                                   "  " + std::to_string((int)std::round(trk_ms)) + " ms", 1);

        cv::imshow("Ball tracking", vis);
        if (vis_writer.isOpened()) vis_writer.write(vis);

        if (cv::waitKey(1) == 27) break;
        frame_idx++;
    }

    cap.release();
    fout.close();
    if (vis_writer.isOpened()) vis_writer.release();
    cv::destroyAllWindows();

    std::cout << "\n=== RESULTS ===\n";
    std::cout << "Total frames             : " << total_frames << "\n";
    std::cout << "GT present frames        : " << gt_present_frames << "\n";
    std::cout << "GT absent frames         : " << gt_absent_frames << "\n";
    std::cout << "Pred present frames      : " << pred_present_frames << "\n";
    std::cout << "Matched (GT&Pred) frames : " << matched_frames << "\n";

    double mean_iou = (matched_frames > 0) ? (sum_iou / matched_frames) : 0.0;
    double mean_center = (matched_frames > 0) ? (sum_center / matched_frames) : 0.0;

    std::cout << "Mean IoU (matched)       : " << mean_iou << "\n";
    std::cout << "Success@IoU>=0.5 (GT)    : " << success_iou_frames
              << " / " << gt_present_frames
              << " = " << (gt_present_frames ? (100.0 * success_iou_frames / gt_present_frames) : 0.0) << "%\n";
    std::cout << "Mean center dist (matched): " << mean_center << " px\n";
    std::cout << "Center<= " << CENTER_OK_PX << "px (GT): " << center_ok_frames
              << " / " << gt_present_frames
              << " = " << (gt_present_frames ? (100.0 * center_ok_frames / gt_present_frames) : 0.0) << "%\n";
    std::cout << "FP on GT-absent frames    : " << fp_on_absent
              << " / " << gt_absent_frames
              << " = " << (gt_absent_frames ? (100.0 * fp_on_absent / gt_absent_frames) : 0.0) << "%\n";

    std::cout << "[INFO] CSV saved: " << csv_out << "\n";
    if (!vis_out.empty()) std::cout << "[INFO] Video saved: " << vis_out << "\n";

    return 0;
}
