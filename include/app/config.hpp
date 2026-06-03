#pragma once

#include "app/tracking_utils.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace app {

struct AppConfig {
    // Paths / inputs
    std::string model_path = "../models/best.onnx";
    std::string video_path = "../data/videos/video_30.avi";   // for video/eval
    std::string xml_path   = "../data/gt/video1.1.xml";       // for eval
    std::string label      = "ball";                          // for eval
    std::string tracker    = "csrt";                          // for eval

    // Outputs
    std::string out_video  = "../data/output/out.avi";        // for video/live record
    std::string out_csv    = "../data/output/eval.csv";       // for eval
    std::string out_vis    = "";                              // for eval optional

    // Live camera
    std::string device     = "/dev/video2";                   // for live
    int req_w = 1280;
    int req_h = 720;
    double record_fps = 30.0;

    Policy policy{};
};

inline std::string trim(std::string s) {
    auto is_ws = [](unsigned char c) { return c==' ' || c=='\t' || c=='\r' || c=='\n'; };
    while (!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && is_ws((unsigned char)s.back()))  s.pop_back();
    return s;
}

inline void mergeInto(std::unordered_map<std::string, std::string>& dst,
                      const std::unordered_map<std::string, std::string>& src) {
    for (const auto& kv : src) dst[kv.first] = kv.second;
}

inline std::unordered_map<std::string, std::string> loadKeyValueFileImpl(
    const std::string& path,
    std::vector<std::string>& include_stack) {

    for (const auto& p : include_stack) {
        if (p == path) {
            std::string msg = "Config include cycle detected:\n";
            for (const auto& x : include_stack) msg += "  -> " + x + "\n";
            msg += "  -> " + path + "\n";
            throw std::runtime_error(msg);
        }
    }
    include_stack.push_back(path);

    std::ifstream fin(path);
    if (!fin.is_open()) {
        include_stack.pop_back();
        throw std::runtime_error("Cannot open cfg file: " + path);
    }

    std::unordered_map<std::string, std::string> merged;
    std::unordered_map<std::string, std::string> local;

    std::string line;
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string k = trim(line.substr(0, eq));
        std::string v = trim(line.substr(eq + 1));
        if (k.empty() || v.empty()) continue;

        if (k == "include") {
            auto inc = loadKeyValueFileImpl(v, include_stack);
            mergeInto(merged, inc);
        } else {
            local[k] = v;
        }
    }

    mergeInto(merged, local);

    include_stack.pop_back();
    return merged;
}

inline std::unordered_map<std::string, std::string> loadKeyValueFile(const std::string& path) {
    std::vector<std::string> stack;
    return loadKeyValueFileImpl(path, stack);
}

inline bool findArgValue(int argc, char** argv, const char* name, std::string& out) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == name) { out = argv[i + 1]; return true; }
    }
    return false;
}

inline void applyPolicy(Policy& p, const std::unordered_map<std::string, std::string>& kv) {
    auto get = [&](const char* key) -> const std::string* {
        auto it = kv.find(key);
        return (it == kv.end()) ? nullptr : &it->second;
    };

    if (auto v = get("redetect_every"))      p.redetect_every = std::stoi(*v);
    if (auto v = get("init_expand"))         p.init_expand = std::stof(*v);
    if (auto v = get("yolo_min_conf"))       p.yolo_min_conf = std::stof(*v);

    if (auto v = get("snap_on_periodic"))    p.snap_on_periodic = (std::stoi(*v) != 0);
    if (auto v = get("snap_center_px"))      p.snap_center_px = std::stod(*v);
    if (auto v = get("snap_iou_min"))        p.snap_iou_min = std::stod(*v);

    if (auto v = get("max_center_jump_px"))  p.max_center_jump_px = std::stod(*v);
    if (auto v = get("min_area_px"))         p.min_area_px = std::stod(*v);
    if (auto v = get("max_area_frac"))       p.max_area_frac = std::stod(*v);
    if (auto v = get("min_inframe_frac"))    p.min_inframe_frac = std::stod(*v);
}

inline void applyAppConfig(AppConfig& c, const std::unordered_map<std::string, std::string>& kv) {
    auto get = [&](const char* key) -> const std::string* {
        auto it = kv.find(key);
        return (it == kv.end()) ? nullptr : &it->second;
    };

    if (auto v = get("model_path")) c.model_path = *v;
    if (auto v = get("video_path")) c.video_path = *v;
    if (auto v = get("xml_path"))   c.xml_path   = *v;
    if (auto v = get("label"))      c.label      = *v;
    if (auto v = get("tracker"))    c.tracker    = *v;

    if (auto v = get("out_video"))  c.out_video = *v;
    if (auto v = get("out_csv"))    c.out_csv   = *v;
    if (auto v = get("out_vis"))    c.out_vis   = *v;

    if (auto v = get("device"))     c.device = *v;
    if (auto v = get("req_w"))      c.req_w = std::stoi(*v);
    if (auto v = get("req_h"))      c.req_h = std::stoi(*v);
    if (auto v = get("record_fps")) c.record_fps = std::stod(*v);

    applyPolicy(c.policy, kv);
}

inline AppConfig loadConfigFromArgsOrDefaults(int argc, char** argv) {
    AppConfig c{};
    std::string cfg_path;
    if (!findArgValue(argc, argv, "--cfg", cfg_path)) return c;

    const auto kv = loadKeyValueFile(cfg_path);
    applyAppConfig(c, kv);
    return c;
}

inline void printConfigBrief(const AppConfig& c) {
    std::cout << "[CFG] model=" << c.model_path << "\n";
    if (!c.video_path.empty()) std::cout << "[CFG] video=" << c.video_path << "\n";
    if (!c.device.empty())     std::cout << "[CFG] device=" << c.device << "\n";
    std::cout << "[CFG] out_video=" << c.out_video << " out_csv=" << c.out_csv << "\n";
    std::cout << "[CFG] redetect=" << c.policy.redetect_every
              << " expand=" << c.policy.init_expand
              << " conf=" << c.policy.yolo_min_conf
              << " snap_px=" << c.policy.snap_center_px
              << " snap_iou=" << c.policy.snap_iou_min
              << " max_jump=" << c.policy.max_center_jump_px
              << "\n";
}
}
