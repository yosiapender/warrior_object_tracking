# WARRIOR_OBJECT_TRACKING

YOLO (ONNX Runtime) + OpenCV tracker hybrid pipeline for ball tracking:

- **Live camera**: `object_tracking`
- **Video input**: `object_tracking_video`
- **Evaluation (CVAT XML)**: `object_tracking_eval`
- **Plots**: `tools/plot_eval.py`

---

## Important Path Rule (Config Files)

All paths in `cfg/*.txt` are resolved relative to the **current working directory**.

Recommended workflow: **always run executables from the project root**.

So config paths should be project-root relative (no `../`), for example:

- `models/best.onnx` ✅
- `data/videos/video_30.avi` ✅
- `../models/best.onnx` ❌

---

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build -j
```

Binaries are written to `./build/`.

---

## Config System

All executables accept:

```bash
--cfg <path_to_config_file>
```

Config format uses simple `key=value` pairs with optional includes:

```txt
include=cfg/policy.txt
model_path=models/model_v3.onnx
video_path=data/videos/challenge1_occluison.avi
out_video=data/output/out.avi
```

Included files are processed first, and local keys override included values.

---

## Example Config Files

### `cfg/policy.txt`

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
tracker=kcf
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

Outputs are written to the paths specified in the configuration files.

---

## Evaluation Plots

`object_tracking_eval` writes CSV containing metrics such as:

- `gt_has`
- `pred_has`
- `iou`
- `center_dist_px`
- `fps`
- additional evaluation metrics

Generate plots with:

```bash
python tools/plot_eval.py --cfg cfg/plot_eval.txt
```

---

## Notes / Troubleshooting

### ONNX model "File doesn't exist"

This usually means that `model_path` is incorrect relative to the current working directory.

Run from the project root and use:

```txt
model_path=models/model_v3.onnx
```

Quick check:

```bash
ls -l models/best.onnx
```

### CUDA vs CPU (ONNX Runtime)

If the ONNX Runtime CUDA provider is available, inference will run on GPU. Otherwise, the detector automatically falls back to CPU execution.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.