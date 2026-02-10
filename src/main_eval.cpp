#include <opencv2/opencv.hpp>

#include "detector/yolo_detector.hpp"
#include "tracker/kcf_tracker.hpp"
#include "tracker/csrt_tracker.hpp"

#include "app/config.hpp"
#include "app/tracking_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr int BOX_THICKNESS  = 2;
constexpr int LINE_STYLE     = cv::LINE_AA;

inline bool pickBestDetection(const std::vector<Detection>& dets, Detection& best) {
    if (dets.empty()) return false;
    best = *std::max_element(
        dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) { return a.score < b.score; });
    return true;
}

// ---- CVAT XML parser helpers ----
inline bool getAttr(const std::string& s, const std::string& key, std::string& out) {
    const std::string pat = key + "=\"";
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
    cv::Rect box{};
};

static std::unordered_map<int, GtFrame> loadCvatXmlTrackBoxes(
    const std::string& xml_path,
    const std::string& target_label) {
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

            const int frame_idx = std::stoi(s_frame);
            const int outside = (!s_outside.empty()) ? std::stoi(s_outside) : 0;

            if (outside == 1) {
                gt[frame_idx] = GtFrame{false, cv::Rect()};
                continue;
            }

            const double xtl = std::stod(s_xtl);
            const double ytl = std::stod(s_ytl);
            const double xbr = std::stod(s_xbr);
            const double ybr = std::stod(s_ybr);

            const int x = static_cast<int>(std::lround(xtl));
            const int y = static_cast<int>(std::lround(ytl));
            const int w = std::max(1, static_cast<int>(std::lround(xbr - xtl)));
            const int h = std::max(1, static_cast<int>(std::lround(ybr - ytl)));

            gt[frame_idx] = GtFrame{true, cv::Rect(x, y, w, h)};
        }
    }
    return gt;
}

// ---- runtime tracker switch ----
struct ITracker {
    virtual ~ITracker() = default;
    virtual bool init(const cv::Mat& frame, const cv::Rect& box) = 0;
    virtual bool update(const cv::Mat& frame, cv::Rect& box) = 0;
    virtual void reset() = 0;
    virtual bool isInitialized() const = 0;
};

struct KcfAdapter final : ITracker {
    KcfTracker t;
    bool init(const cv::Mat& f, const cv::Rect& b) override { return t.init(f, b); }
    bool update(const cv::Mat& f, cv::Rect& b) override { return t.update(f, b); }
    void reset() override { t.reset(); }
    bool isInitialized() const override { return t.isInitialized(); }
};

struct CsrtAdapter final : ITracker {
    CsrtTracker t;
    bool init(const cv::Mat& f, const cv::Rect& b) override { return t.init(f, b); }
    bool update(const cv::Mat& f, cv::Rect& b) override { return t.update(f, b); }
    void reset() override { t.reset(); }
    bool isInitialized() const override { return t.isInitialized(); }
};

} // namespace

int main(int argc, char** argv) {
    const app::AppConfig cfg = app::loadConfigFromArgsOrDefaults(argc, argv);
    app::printConfigBrief(cfg);

    const std::string onnx_path  = cfg.model_path;
    const std::string video_path = cfg.video_path;
    const std::string xml_path   = cfg.xml_path;
    const std::string label      = cfg.label;
    const std::string tracker_ty = cfg.tracker;
    const std::string csv_out    = cfg.out_csv;
    const std::string vis_out    = cfg.out_vis;
    const app::Policy  policy    = cfg.policy;

    // evaluation thresholds (kept as constants for now)
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

    // Optional visualization writer
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

    // Tracker switch
    std::unique_ptr<ITracker> tracker;
    if (tracker_ty == "kcf") tracker = std::make_unique<KcfAdapter>();
    else if (tracker_ty == "csrt") tracker = std::make_unique<CsrtAdapter>();
    else {
        std::cerr << "[ERROR] Unknown tracker type: " << tracker_ty << " (use kcf|csrt)\n";
        return -1;
    }

    cv::Rect track_box{};
    bool have_track = false;

    app::Timer total_t, yolo_t, trk_t;

    int frame_idx = 0;
    cv::Rect prev_box{};
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

        ++total_frames;
        total_t.start();

        // ---- GT ----
        bool gt_has = false;
        cv::Rect gt_box{};
        if (auto it = gt.find(frame_idx); it != gt.end() && it->second.has) {
            gt_has = true;
            gt_box = app::clampRect(it->second.box, frame.size());
        }
        if (gt_has) ++gt_present_frames;
        else ++gt_absent_frames;

        const bool periodic_redetect =
            (policy.redetect_every > 0 && (frame_idx % policy.redetect_every == 0));
        const bool need_yolo =
            !tracker->isInitialized() || !have_track || periodic_redetect;

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

            const bool must_init = !tracker->isInitialized() || !have_track;
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
                tracker->reset();
                if (tracker->init(frame, init_box)) {
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
        if (!reinit && tracker->isInitialized() && have_track) {
            trk_t.start();
            cv::Rect new_box = track_box;
            bool ok = tracker->update(frame, new_box);
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
                tracker->reset();
            }
        }

        const bool pred_has = have_track && tracker->isInitialized();
        if (pred_has) ++pred_present_frames;

        // ---- Eval metrics ----
        double iou = -1.0;
        double cd  = -1.0;

        if (gt_has && pred_has) {
            ++matched_frames;
            iou = app::IoU(gt_box, track_box);
            cd  = cv::norm(app::rectCenter(gt_box) - app::rectCenter(track_box));
            sum_iou += iou;
            sum_center += cd;
            if (iou >= IOU_SUCCESS_THR) ++success_iou_frames;
            if (cd <= CENTER_OK_PX) ++center_ok_frames;
        } else if (!gt_has && pred_has) {
            ++fp_on_absent;
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
            cv::rectangle(vis, gt_box, cv::Scalar(0, 255, 255), BOX_THICKNESS, LINE_STYLE);
        }
        if (pred_has) {
            cv::rectangle(vis, track_box, cv::Scalar(255, 0, 0), BOX_THICKNESS, LINE_STYLE);
        } else {
            cv::putText(vis, "TRACK LOST",
                        {10, 90}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 0, 255), 2, LINE_STYLE);
        }

        const double total_ms = total_t.stopMs();
        const double fps = (total_ms > 0.0) ? (1000.0 / total_ms) : 0.0;

        cv::putText(vis, "FPS: " + std::to_string(static_cast<int>(fps)),
                    {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2, LINE_STYLE);

        cv::putText(vis, "Yellow=GT  Blue=YOLO+TRK",
                    {10, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 255), 2, LINE_STYLE);

        app::drawTopRightStatus(vis,
                                std::string("YOLO: ") + (yolo_has ? "ON" : "OFF") +
                                    "  " + std::to_string(static_cast<int>(std::lround(yolo_ms))) + " ms",
                                0);
        app::drawTopRightStatus(vis,
                                std::string("TRK: ") + (pred_has ? "ON" : "OFF") +
                                    "  " + std::to_string(static_cast<int>(std::lround(trk_ms))) + " ms",
                                1);

        cv::imshow("Ball tracking", vis);
        if (vis_writer.isOpened()) vis_writer.write(vis);

        if (cv::waitKey(1) == 27) break;
        ++frame_idx;
    }

    cap.release();
    fout.close();
    if (vis_writer.isOpened()) vis_writer.release();
    cv::destroyAllWindows();

    std::cout << "\n=== RESULTS ===\n";
    std::cout << "Total frames              : " << total_frames << "\n";
    std::cout << "GT present frames         : " << gt_present_frames << "\n";
    std::cout << "GT absent frames          : " << gt_absent_frames << "\n";
    std::cout << "Pred present frames       : " << pred_present_frames << "\n";
    std::cout << "Matched (GT&Pred) frames  : " << matched_frames << "\n";

    const double mean_iou = (matched_frames > 0) ? (sum_iou / matched_frames) : 0.0;
    const double mean_center = (matched_frames > 0) ? (sum_center / matched_frames) : 0.0;

    std::cout << "Mean IoU (matched)        : " << mean_iou << "\n";
    std::cout << "Success@IoU>=0.5 (GT)     : " << success_iou_frames
              << " / " << gt_present_frames
              << " = " << (gt_present_frames ? (100.0 * success_iou_frames / gt_present_frames) : 0.0) << "%\n";
    std::cout << "Mean center dist (matched): " << mean_center << " px\n";
    std::cout << "Center<= " << CENTER_OK_PX << "px (GT): " << center_ok_frames
              << " / " << gt_present_frames
              << " = " << (gt_present_frames ? (100.0 * center_ok_frames / gt_present_frames) : 0.0) << "%\n";
    std::cout << "FP on GT-absent frames     : " << fp_on_absent
              << " / " << gt_absent_frames
              << " = " << (gt_absent_frames ? (100.0 * fp_on_absent / gt_absent_frames) : 0.0) << "%\n";

    std::cout << "[INFO] CSV saved: " << csv_out << "\n";
    if (!vis_out.empty()) std::cout << "[INFO] Video saved: " << vis_out << "\n";
    return 0;
}
