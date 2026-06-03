# WARRIOR_OBJECT_TRACKING

YOLO (ONNX Runtime) + OpenCV tracker (KCF/CSRT) hybrid pipeline for ball tracking:
- **Live camera**: `object_tracking`
- **Video input**: `object_tracking_video`
- **Evaluation (CVAT XML)**: `object_tracking_eval`
- **Plots**: `tools/plot_eval.py`

---

## Project Layout

```
WARRIOR_OBJECT_TRACKING/
├── cfg/
│   ├── plot_eval.txt
│   ├── policy.txt
│   ├── run_live.txt
│   ├── run_video.txt
│   └── run_eval.txt
├── data/
│   ├── gt/
│   ├── images/
│   ├── output/
│   └── videos/
├── include/
│   ├── app/
│   │   ├── config.hpp
│   │   └── tracking_utils.hpp
│   ├── detector/
│   │   └── yolo_detector.hpp
│   └── tracker/
│       ├── colornames_lut.hpp
│       ├── csrt_tracker.hpp
│       ├── kcf_local.hpp
│       └── kcf_tracker.hpp
├── models/
│   ├── best.onnx
│   ├── model_v2.onnx
│   └── model_v3.onnx
├── src/
│   ├── detector/
│   │   └── yolo_detector.cpp
│   ├── tracker/
│       ├── colornames_lut.cpp
│       ├── csrt_tracker.cpp
│       ├── kcf_local.cpp
│       └── kcf_tracker.cpp
│   ├── main_eval.cpp
│   ├── main_video.cpp
│   └── main.cpp
├── tools/
│   └── plot_eval.py
└── CMakeLists.txt
```

---

## Important Path Rule (Config Files)

All paths in `cfg/*.txt` are resolved relative to the **current working directory**.

Recommended workflow: **always run executables from the project root**  
So config paths should be **project-root relative** (no `../`), e.g.:

- `models/best.onnx` ✅
- `data/videos/video_30.avi` ✅
- `../models/best.onnx` ❌ (will point outside the project)

---

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build -j
```

Binaries are written to `./build/` (as configured in `CMakeLists.txt`).

---

## Config System

All executables accept:

```bash
--cfg <path_to_config_file>
```

Config format is simple `key=value` with optional includes:

```txt
include=cfg/policy.txt
model_path=models/model_v3.onnx
video_path=data/videos/challenge1_occluison.avi
out_video=data/output/out.avi
```

Includes are processed first, local keys override included keys. Cycles are detected.

---

## Example Config Files

### `cfg/policy.txt`
Shared tracking policy knobs:

```txt
redetect_every=15
init_expand=1.25
yolo_min_conf=0.25
snap_on_periodic=1
snap_center_px=30
snap_iou_min=0.20
max_center_jump_px=120
min_area_px=16
max_area_frac=0.35
min_inframe_frac=0.65
```

### `cfg/run_video.txt`
```txt
include=cfg/policy.txt

model_path=models/model_v3.onnx
video_path=data/videos/video_30.avi
out_video=data/output/no_obj.avi
```

### `cfg/run_live.txt`
```txt
include=cfg/policy.txt

model_path=models/model_v3.onnx
device=/dev/video2
req_w=1280
req_h=720
record_fps=30
out_video=data/output/live.avi
```

### `cfg/run_eval.txt`
```txt
include=cfg/policy.txt

model_path=models/model_v3.onnx
video_path=data/videos/1_occlusion.avi
xml_path=data/gt/video1.1.xml
label=ball
tracker=csrt
out_csv=data/output/eval_hybrid.csv
out_vis=data/output/eval_vis.avi
```

---

## Run

### Video
```bash
./build/object_tracking_video --cfg cfg/run_video.txt
```

### Live Camera
```bash
./build/object_tracking --cfg cfg/run_live.txt
```

### Evaluation (CVAT XML)
```bash
./build/object_tracking_eval --cfg cfg/run_eval.txt
```

Outputs are written to the paths specified in the config (typically under `data/output/`).

---

## Evaluation Plots

`object_tracking_eval` writes CSV with:
- `gt_has`, `pred_has`, `iou`, `center_dist_px`, etc.

Then plot:

```bash
python tools/plot_eval.py --csv data/output/eval_hybrid.csv --out_prefix data/output/eval_matched
```

Optional knobs:

```bash
python tools/plot_eval.py --csv data/output/eval_hybrid.csv --max_dist 30 --p_at 20
```

---

## Notes / Troubleshooting

### ONNX model “File doesn’t exist”
This usually means your `model_path` is wrong relative to where you run the command.

Run from project root and use `model_path=models/model_v3.onnx`.

Quick check:

```bash
ls -l models/best.onnx
```

### CUDA vs CPU (ONNX Runtime)
If ONNX Runtime CUDA provider is available, the detector will attempt to use it; otherwise it falls back to CPU (your code prints a warning).

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
