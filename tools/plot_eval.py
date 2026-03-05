#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Set, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Small utilities
# ----------------------------
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _fmt_ratio(n: int, d: int) -> str:
    if d <= 0:
        return f"{n} / {d} = 0.000"
    return f"{n} / {d} = {n / d:.3f}"


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    trap = getattr(np, "trapezoid", None)
    if trap is None:
        trap = np.trapz
    return float(trap(y, x))


def _enforce_monotonic(y: np.ndarray, increasing: bool) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0.0, 1.0)
    if y.size == 0:
        return y
    if increasing:
        return np.maximum.accumulate(y)
    return np.minimum.accumulate(y)


def _smooth_ma(y: np.ndarray, win: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if win <= 1 or y.size < 3:
        return y
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    k = np.ones(win, dtype=float) / float(win)
    return np.convolve(yp, k, mode="valid")


def _wrap_lines(s: str, width: int = 44, max_lines: int = 2) -> str:
    """
    Wrap on spaces into <= max_lines (no ellipsis).
    If still too long, it will simply keep the first max_lines lines.
    """
    s = (s or "").strip()
    if not s:
        return ""
    words = s.split()
    lines = []
    cur = ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if len(nxt) <= width:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = w
            if len(lines) >= max_lines - 1:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return "\n".join(lines[:max_lines])


# ----------------------------
# KV config loader (cfg)
# ----------------------------
def _read_kv_config(path: Path, _stack: Optional[Set[Path]] = None) -> Dict[str, str]:
    path = path.expanduser().resolve()
    if _stack is None:
        _stack = set()
    if path in _stack:
        raise ValueError(f"Config include cycle detected at: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    _stack.add(path)
    out: Dict[str, str] = {}

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()

        if not k:
            continue

        if k.lower() == "include":
            inc = (path.parent / v).expanduser().resolve()
            out.update(_read_kv_config(inc, _stack))
        else:
            out[k] = v

    _stack.remove(path)
    return out


def _apply_cfg_defaults(parser: argparse.ArgumentParser, cfg_path: Path) -> None:
    raw = _read_kv_config(cfg_path)

    dest_to_action = {}
    for a in parser._actions:
        if a.dest and a.dest != "help":
            dest_to_action[a.dest] = a

    defaults: Dict[str, object] = {}
    unknown = []

    for key, sval in raw.items():
        dest = key.strip()
        if dest not in dest_to_action:
            unknown.append(dest)
            continue

        action = dest_to_action[dest]
        if action.type is not None:
            val = action.type(sval)
        else:
            d = action.default
            if isinstance(d, bool):
                val = sval.lower() in ("1", "true", "yes", "on")
            elif isinstance(d, int):
                val = int(sval)
            elif isinstance(d, float):
                val = float(sval)
            else:
                val = sval

        if action.choices is not None and val not in action.choices:
            raise ValueError(f"Invalid value for '{dest}': {val} (choices={list(action.choices)})")

        defaults[dest] = val

    if unknown:
        raise ValueError(f"Unknown config keys in {cfg_path}: {unknown}")

    parser.set_defaults(**defaults)


# ----------------------------
# Smooth curve builders
# ----------------------------
def _gaussian_kernel1d(sigma_bins: float, max_len: int) -> np.ndarray:
    sigma_bins = float(sigma_bins)
    if sigma_bins <= 0 or max_len <= 1:
        return np.array([1.0], dtype=float)

    r = int(np.ceil(3.0 * sigma_bins))
    r_cap = (int(max_len) - 1) // 2
    r = max(1, min(r, r_cap))

    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(x * x) / (2.0 * sigma_bins * sigma_bins))
    k /= k.sum()
    return k


def _smooth_cdf_from_samples(
    samples: np.ndarray,
    x_grid: np.ndarray,
    domain_lo: float,
    domain_hi: float,
    bins: int,
    sigma: float,
    force_start: float | None,
    force_end: float | None,
    increasing: bool,
) -> np.ndarray:
    x_grid = np.asarray(x_grid, dtype=float)
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        y = np.zeros_like(x_grid)
        if not increasing:
            y = np.ones_like(x_grid)
        return y

    lo = float(domain_lo)
    hi = float(domain_hi)
    if hi <= lo:
        raise ValueError("Invalid domain for KDE smoothing")

    s = np.clip(samples, lo, hi)
    nb = max(64, int(bins))

    hist, edges = np.histogram(s, bins=nb, range=(lo, hi), density=False)
    hist = hist.astype(float)

    if float(sigma) > 0:
        bin_w = (hi - lo) / nb
        sigma_bins = float(sigma) / bin_w
        k = _gaussian_kernel1d(sigma_bins=sigma_bins, max_len=nb)
        hist = np.convolve(hist, k, mode="same")

    total = hist.sum()
    if total <= 0:
        cdf_centers = np.zeros(nb, dtype=float)
    else:
        pdf = hist / total
        cdf_centers = np.cumsum(pdf)
        cdf_centers = np.clip(cdf_centers, 0.0, 1.0)

    centers = 0.5 * (edges[:-1] + edges[1:])
    cdf_on_grid = np.interp(x_grid, centers, cdf_centers, left=cdf_centers[0], right=cdf_centers[-1])
    cdf_on_grid = _enforce_monotonic(cdf_on_grid, increasing=True)

    if increasing:
        y = cdf_on_grid
        if force_start is not None:
            y[0] = float(force_start)
        if force_end is not None:
            y[-1] = float(force_end)
        return _enforce_monotonic(y, increasing=True)

    y = 1.0 - cdf_on_grid
    if force_start is not None:
        y[0] = float(force_start)
    if force_end is not None:
        y[-1] = float(force_end)
    return _enforce_monotonic(y, increasing=False)


def _smooth_monotone_interpolated(
    x: np.ndarray,
    y: np.ndarray,
    x_dense: np.ndarray,
    increasing: bool,
    anchor_points: int,
    smooth_win: int,
    force_start: float | None,
    force_end: float | None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_dense = np.asarray(x_dense, dtype=float)

    y = _enforce_monotonic(y, increasing=increasing)

    if x.size < 3:
        yd = np.interp(x_dense, x, y)
        if force_start is not None:
            yd[0] = float(force_start)
        if force_end is not None:
            yd[-1] = float(force_end)
        return _enforce_monotonic(yd, increasing=increasing)

    anchor_points = max(16, int(anchor_points))
    xa = np.linspace(float(x[0]), float(x[-1]), anchor_points, dtype=float)
    ya = np.interp(xa, x, y)

    if smooth_win and smooth_win > 1:
        ya = _smooth_ma(ya, int(smooth_win))

    ya = _enforce_monotonic(ya, increasing=increasing)

    if force_start is not None:
        ya[0] = float(force_start)
    if force_end is not None:
        ya[-1] = float(force_end)
    ya = _enforce_monotonic(ya, increasing=increasing)

    yd = np.interp(x_dense, xa, ya)
    if force_start is not None:
        yd[0] = float(force_start)
    if force_end is not None:
        yd[-1] = float(force_end)
    return _enforce_monotonic(yd, increasing=increasing)


# ----------------------------
# Core computation helpers
# ----------------------------
def _compute_curves(
    iou: np.ndarray,
    cd: np.ndarray,
    gt_present: int,
    dist_thrs: np.ndarray,
    iou_thrs: np.ndarray,
):
    precision_cond = np.array([(cd <= d).mean() for d in dist_thrs], dtype=float)
    success_cond = np.array([(iou >= t).mean() for t in iou_thrs], dtype=float)

    if gt_present > 0:
        precision_overall = np.array([((cd <= d).sum() / gt_present) for d in dist_thrs], dtype=float)
        success_overall = np.array([((iou >= t).sum() / gt_present) for t in iou_thrs], dtype=float)
    else:
        precision_overall = np.zeros_like(dist_thrs, dtype=float)
        success_overall = np.zeros_like(iou_thrs, dtype=float)

    return precision_cond, precision_overall, success_cond, success_overall


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--cfg", type=str, default=None,
                    help="Optional key=value config file to set defaults (CLI overrides).")

    ap.add_argument("--csv", default=None,
                    help="CSV from main_eval.cpp (can be set in --cfg as csv=...).")
    ap.add_argument("--out_prefix", default="eval",
                    help="Output prefix (path without extension).")

    ap.add_argument("--max_dist", type=float, default=50.0,
                    help="Max distance (px) for precision curve x-axis.")
    ap.add_argument("--curve_points", type=int, default=2001,
                    help="Number of points for smoothed curves.")

    ap.add_argument("--smooth_method", choices=["none", "anchor", "kde"], default="kde",
                    help="Smoothing method for curves.")
    ap.add_argument("--smooth_win", type=int, default=15,
                    help="Window size for 'anchor' smoothing.")
    ap.add_argument("--anchor_points", type=int, default=151,
                    help="Number of anchor points for 'anchor' smoothing.")

    ap.add_argument("--kde_bins", type=int, default=256,
                    help="Bins for KDE-based smoothing.")
    ap.add_argument("--kde_sigma_px", type=float, default=0.8,
                    help="Gaussian sigma (px) for KDE smoothing in precision.")
    ap.add_argument("--kde_sigma_iou", type=float, default=0.02,
                    help="Gaussian sigma (IoU) for KDE smoothing in success.")

    ap.add_argument("--p_at", type=float, default=20.0,
                    help="Precision@X threshold shown on plot.")
    ap.add_argument("--success_iou", type=float, default=0.5,
                    help="Success threshold IoU shown on plot.")

    ap.add_argument("--timeline_stride", type=int, default=1,
                    help="Plot every Nth frame on the timeline.")

    ap.add_argument("--condition", type=str, default="",
                    help="Condition name shown in titles, e.g., 'Fast motion', 'Occlusion'.")
    ap.add_argument("--title_mode", choices=["precision", "success", "both"], default="both",
                    help="Where to apply the condition title.")

    ap.add_argument("--tracker", type=str, default="KCF",
                    help="Base tracker name used in legend (e.g., KCF, CSRT).")
    ap.add_argument("--legend_mode", choices=["auto", "kcf", "kcf+yolo"], default="auto",
                    help="Legend label mode. auto uses yolo_has>0 to decide.")

    # parse cfg first, then parse full args
    pre, _ = ap.parse_known_args()
    if pre.cfg is not None:
        _apply_cfg_defaults(ap, Path(pre.cfg))
    args = ap.parse_args()

    if not args.csv:
        ap.error("Missing CSV input. Provide --csv <file> or put 'csv = <file>' in --cfg.")

    df = pd.read_csv(args.csv)

    required = {"frame", "gt_has", "pred_has", "iou", "center_dist_px", "yolo_has", "reinit"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns {missing}. Found: {list(df.columns)}")

    df = df.sort_values("frame").reset_index(drop=True)

    for col in ["frame", "gt_has", "pred_has", "yolo_has", "reinit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["iou", "center_dist_px"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----------------------------
    # Titles (short + wrapped, no truncation)
    # ----------------------------
    cond = str(args.condition).strip()

    def _title_for(mode: str, fallback: str) -> str:
        if not cond:
            return fallback
        if args.title_mode == "both" or args.title_mode == mode:
            return _wrap_lines(cond, width=44, max_lines=2)
        return fallback

    def _timeline_title(metric: str) -> str:
        # keep it short so it never gets cut
        base = f"{metric} timeline (matched only)"
        if not cond:
            return base
        # 2 lines: condition on line1, short base on line2
        return f"{_wrap_lines(cond, width=44, max_lines=1)}\n{base}"

    # ----------------------------
    # Counts
    # ----------------------------
    total_frames = int(len(df))
    gt_present = int((df["gt_has"] == 1).sum())
    gt_absent = total_frames - gt_present

    pred_present = int((df["pred_has"] == 1).sum())
    matched_df = df[(df["gt_has"] == 1) & (df["pred_has"] == 1)].copy()
    matched_frames = int(len(matched_df))

    misses_gt_present = int(((df["gt_has"] == 1) & (df["pred_has"] == 0)).sum())
    fp_gt_absent = int(((df["gt_has"] == 0) & (df["pred_has"] == 1)).sum())

    coverage = (matched_frames / gt_present) if gt_present else 0.0
    yolo_on = int((df["yolo_has"] == 1).sum())
    reinit_n = int((df["reinit"] == 1).sum())

    # ----------------------------
    # Legend label (KCF vs KCF+YOLO)
    # ----------------------------
    base = str(args.tracker).strip() or "KCF"
    if args.legend_mode == "kcf":
        tracker_label = base
    elif args.legend_mode == "kcf+yolo":
        tracker_label = f"{base}+YOLO"
    else:
        tracker_label = f"{base}+YOLO" if yolo_on > 0 else base

    # matched-only samples
    m = matched_df[np.isfinite(matched_df["iou"]) & np.isfinite(matched_df["center_dist_px"])].copy()
    if m.empty:
        raise SystemExit("No valid matched frames with finite IoU and center distance.")

    iou = m["iou"].to_numpy(dtype=float)
    cd = m["center_dist_px"].to_numpy(dtype=float)

    n = max(301, int(args.curve_points))
    dist_thrs = np.linspace(0.0, float(args.max_dist), n, dtype=float)
    iou_thrs = np.linspace(0.0, 1.0, n, dtype=float)

    prec_cond, prec_overall, succ_cond, succ_overall = _compute_curves(
        iou=iou,
        cd=cd,
        gt_present=gt_present,
        dist_thrs=dist_thrs,
        iou_thrs=iou_thrs,
    )

    prec_cond = _enforce_monotonic(prec_cond, increasing=True)
    prec_overall = _enforce_monotonic(prec_overall, increasing=True)
    succ_cond = _enforce_monotonic(succ_cond, increasing=False)
    succ_overall = _enforce_monotonic(succ_overall, increasing=False)

    method = args.smooth_method

    force_prec_start = 0.0
    force_prec_end_cond = float(prec_cond[-1])
    force_prec_end_overall = float(prec_overall[-1])

    force_succ_start_cond = float(succ_cond[0])
    force_succ_end_cond = float(succ_cond[-1])
    force_succ_start_overall = float(succ_overall[0])
    force_succ_end_overall = float(succ_overall[-1])

    if method == "none":
        pass
    elif method == "anchor":
        apoints = int(args.anchor_points)
        swin = int(args.smooth_win)

        prec_cond = _smooth_monotone_interpolated(
            dist_thrs, prec_cond, dist_thrs, True, apoints, swin, force_prec_start, force_prec_end_cond
        )
        prec_overall = _smooth_monotone_interpolated(
            dist_thrs, prec_overall, dist_thrs, True, apoints, swin, force_prec_start, force_prec_end_overall
        )

        succ_cond = _smooth_monotone_interpolated(
            iou_thrs, succ_cond, iou_thrs, False, apoints, swin, force_succ_start_cond, force_succ_end_cond
        )
        succ_overall = _smooth_monotone_interpolated(
            iou_thrs, succ_overall, iou_thrs, False, apoints, swin, force_succ_start_overall, force_succ_end_overall
        )
    elif method == "kde":
        bins = int(args.kde_bins)
        sigma_px = float(args.kde_sigma_px)

        prec_cond = _smooth_cdf_from_samples(
            samples=cd,
            x_grid=dist_thrs,
            domain_lo=0.0,
            domain_hi=float(args.max_dist),
            bins=bins,
            sigma=sigma_px,
            force_start=force_prec_start,
            force_end=force_prec_end_cond,
            increasing=True,
        )

        prec_overall_tmp = _smooth_cdf_from_samples(
            samples=cd,
            x_grid=dist_thrs,
            domain_lo=0.0,
            domain_hi=float(args.max_dist),
            bins=bins,
            sigma=sigma_px,
            force_start=force_prec_start,
            force_end=None,
            increasing=True,
        )
        if prec_overall_tmp[-1] > 1e-12:
            scale = force_prec_end_overall / prec_overall_tmp[-1]
            prec_overall = np.clip(prec_overall_tmp * scale, 0.0, 1.0)
        else:
            prec_overall = np.zeros_like(prec_overall_tmp)
        prec_overall[0] = force_prec_start
        prec_overall[-1] = force_prec_end_overall
        prec_overall = _enforce_monotonic(prec_overall, increasing=True)

        sigma_iou = float(args.kde_sigma_iou)

        succ_cond = _smooth_cdf_from_samples(
            samples=iou,
            x_grid=iou_thrs,
            domain_lo=0.0,
            domain_hi=1.0,
            bins=bins,
            sigma=sigma_iou,
            force_start=None,
            force_end=None,
            increasing=False,
        )
        succ_cond[0] = force_succ_start_cond
        succ_cond[-1] = force_succ_end_cond
        succ_cond = _enforce_monotonic(succ_cond, increasing=False)

        succ_overall_tmp = _smooth_cdf_from_samples(
            samples=iou,
            x_grid=iou_thrs,
            domain_lo=0.0,
            domain_hi=1.0,
            bins=bins,
            sigma=sigma_iou,
            force_start=None,
            force_end=None,
            increasing=False,
        )
        target0 = force_succ_start_overall
        cur0 = float(succ_overall_tmp[0])
        if cur0 > 1e-12:
            scale = target0 / cur0
            succ_overall = np.clip(succ_overall_tmp * scale, 0.0, 1.0)
        else:
            succ_overall = np.zeros_like(succ_overall_tmp)
        succ_overall[0] = force_succ_start_overall
        succ_overall[-1] = force_succ_end_overall
        succ_overall = _enforce_monotonic(succ_overall, increasing=False)

    # ----------------------------
    # Point metrics
    # ----------------------------
    p_at = float(args.p_at)
    thr = float(args.success_iou)

    p_at_cond = float((cd <= p_at).mean())
    succ_count = int((iou >= thr).sum())
    succ_cond_at = float((succ_count / matched_frames) if matched_frames else 0.0)

    p_at_overall = float(((cd <= p_at).sum() / gt_present) if gt_present else 0.0)
    succ_overall_at = float((succ_count / gt_present) if gt_present else 0.0)

    auc_cond = _trapz(succ_cond, iou_thrs) / (iou_thrs[-1] - iou_thrs[0])
    auc_overall = (_trapz(succ_overall, iou_thrs) / (iou_thrs[-1] - iou_thrs[0])) if gt_present else 0.0
    effective_auc = float(auc_cond * coverage)

    # ----------------------------
    # Plot styling
    # ----------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "lines.linewidth": 3.0,
        "figure.dpi": 120,
        "savefig.dpi": 250,
    })

    out_prefix = Path(args.out_prefix)
    _ensure_parent(out_prefix.with_suffix(".dummy"))

    # ----------------------------
    # OTB-like combined figure (2 panels)
    # ----------------------------
    plt.figure(figsize=(14.5, 5.8))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.plot(
        dist_thrs, prec_overall,
        label=f"{tracker_label}  P@{p_at:g}px={p_at_overall:.3f}  Cov={coverage:.3f}",
        solid_joinstyle="round", solid_capstyle="round", antialiased=True
    )
    ax1.set_xlim(0, args.max_dist)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("Location error threshold")
    ax1.set_ylabel("Precision rate")
    ax1.set_title(_title_for("precision", "Precision plots of OPE"))
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.grid(False)
    ax1.tick_params(direction="in")

    ax2.plot(
        iou_thrs, succ_overall,
        label=f"{tracker_label}  AUC={auc_overall:.3f}  S@{thr:g}={succ_overall_at:.3f}",
        solid_joinstyle="round", solid_capstyle="round", antialiased=True
    )
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("Overlap threshold")
    ax2.set_ylabel("Success rate")
    ax2.set_title(_title_for("success", "Success plots of OPE"))
    ax2.legend(loc="lower left", framealpha=0.95)
    ax2.grid(False)
    ax2.tick_params(direction="in")

    plt.tight_layout()
    otb_png = str(out_prefix) + "_otb.png"
    plt.savefig(otb_png)
    plt.close()

    # ----------------------------
    # Precision plots
    # ----------------------------
    plt.figure(figsize=(8.2, 5.8))
    plt.plot(
        dist_thrs, prec_cond,
        label=f"{tracker_label} (Conditional)  P@{p_at:g}px={p_at_cond:.3f}",
        solid_joinstyle="round", solid_capstyle="round", antialiased=True
    )
    plt.xlim(0, args.max_dist)
    plt.ylim(0, 1.0)
    plt.xlabel("Location error threshold (pixels)")
    plt.ylabel("Precision rate")
    plt.title(_title_for("precision", "Precision (Conditional / matched)"))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    prec_cond_png = str(out_prefix) + "_precision_conditional.png"
    plt.savefig(prec_cond_png)
    plt.close()

    plt.figure(figsize=(8.2, 5.8))
    plt.plot(
        dist_thrs, prec_overall,
        label=f"{tracker_label} (Overall)  P@{p_at:g}px={p_at_overall:.3f}  Cov={coverage:.3f}",
        solid_joinstyle="round", solid_capstyle="round", antialiased=True
    )
    plt.xlim(0, args.max_dist)
    plt.ylim(0, 1.0)
    plt.xlabel("Location error threshold (pixels)")
    plt.ylabel("Precision rate")
    plt.title(_title_for("precision", "Precision (Overall / GT-present)"))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    prec_overall_png = str(out_prefix) + "_precision_overall.png"
    plt.savefig(prec_overall_png)
    plt.close()

    plt.figure(figsize=(8.2, 5.8))
    plt.plot(
        dist_thrs, prec_cond,
        label=f"{tracker_label} Conditional  P@{p_at:g}px={p_at_cond:.3f}",
        solid_joinstyle="round", solid_capstyle="round", antialiased=True
    )
    plt.plot(
        dist_thrs, prec_overall,
        label=f"{tracker_label} Overall  P@{p_at:g}px={p_at_overall:.3f}  Cov={coverage:.3f}",
        solid_joinstyle="round", solid_capstyle="round", antialiased=True
    )
    plt.xlim(0, args.max_dist)
    plt.ylim(0, 1.0)
    plt.xlabel("Location error threshold (pixels)")
    plt.ylabel("Precision rate")
    plt.title(_title_for("precision", "Precision — Conditional vs Overall"))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    prec_both_png = str(out_prefix) + "_precision_both.png"
    plt.savefig(prec_both_png)
    plt.close()

    # ----------------------------
    # Success plots
    # ----------------------------
    plt.figure(figsize=(8.2, 5.8))
    plt.plot(
        iou_thrs, succ_cond,
        label=f"{tracker_label} (Conditional)  AUC={auc_cond:.3f}  S@{thr:g}={succ_cond_at:.3f}",
        solid_joinstyle="round", solid_capstyle="round", antialiased=True
    )
    plt.xlim(0.0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Overlap threshold (IoU)")
    plt.ylabel("Success rate")
    plt.title(_title_for("success", "Success (Conditional / matched)"))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    succ_cond_png = str(out_prefix) + "_success_conditional.png"
    plt.savefig(succ_cond_png)
    plt.close()

    plt.figure(figsize=(8.2, 5.8))
    plt.plot(
        iou_thrs, succ_overall,
        label=f"{tracker_label} (Overall)  AUC={auc_overall:.3f}  S@{thr:g}={succ_overall_at:.3f}  Cov={coverage:.3f}",
        solid_joinstyle="round", solid_capstyle="round", antialiased=True
    )
    plt.xlim(0.0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Overlap threshold (IoU)")
    plt.ylabel("Success rate")
    plt.title(_title_for("success", "Success (Overall / GT-present)"))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    succ_overall_png = str(out_prefix) + "_success_overall.png"
    plt.savefig(succ_overall_png)
    plt.close()

    plt.figure(figsize=(8.2, 5.8))
    plt.plot(
        iou_thrs, succ_cond,
        label=f"{tracker_label} Conditional  AUC={auc_cond:.3f}",
        solid_joinstyle="round", solid_capstyle="round", antialiased=True
    )
    plt.plot(
        iou_thrs, succ_overall,
        label=f"{tracker_label} Overall  AUC={auc_overall:.3f}  Cov={coverage:.3f}",
        solid_joinstyle="round", solid_capstyle="round", antialiased=True
    )
    plt.xlim(0.0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Overlap threshold (IoU)")
    plt.ylabel("Success rate")
    plt.title(_title_for("success", "Success — Conditional vs Overall"))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    succ_both_png = str(out_prefix) + "_success_both.png"
    plt.savefig(succ_both_png)
    plt.close()

    # ----------------------------
    # Timeline plots (FIXED axis + tidy title + better legend)
    # ----------------------------
    # ----------------------------
    # Timeline plots (x-axis to last frame, draw ONLY matched points)
    # ----------------------------
    stride = max(1, int(args.timeline_stride))
    dft = df.iloc[::stride].copy()

    frames = dft["frame"].to_numpy(dtype=int)

    gt_t = dft["gt_has"].to_numpy(dtype=int)
    pr_t = dft["pred_has"].to_numpy(dtype=int)

    iou_t = dft["iou"].to_numpy(dtype=float)
    cd_t = dft["center_dist_px"].to_numpy(dtype=float)

    matched_mask = (gt_t == 1) & (pr_t == 1) & np.isfinite(iou_t) & np.isfinite(cd_t)

    # Matched-only x/y (no NaN gaps -> nothing drawn when no match)
    x_m = frames[matched_mask]
    iou_m = iou_t[matched_mask]
    cd_m = cd_t[matched_mask]

    iou_mean = float(iou_m.mean()) if iou_m.size else float("nan")
    cd_mean = float(cd_m.mean()) if cd_m.size else float("nan")

    reinit_f = dft.loc[dft["reinit"] == 1, "frame"].to_numpy(dtype=int)
    yolo_f = dft.loc[dft["yolo_has"] == 1, "frame"].to_numpy(dtype=int)

    x_min = int(frames.min()) if frames.size else 0
    x_max = int(frames.max()) if frames.size else 0

    # ----- IoU timeline -----
    plt.figure(figsize=(10.5, 4.6))
    if x_m.size:
        plt.plot(x_m, iou_m, label="IoU (matched)")
    if np.isfinite(iou_mean):
        plt.axhline(iou_mean, color="tab:red", linestyle="--", linewidth=2.0,
                    label=f"Mean IoU={iou_mean:.3f}")

    if reinit_f.size:
        plt.scatter(reinit_f, np.full_like(reinit_f, 1.02, dtype=float), marker="|", label="reinit")
    if yolo_f.size:
        plt.scatter(yolo_f, np.full_like(yolo_f, 1.08, dtype=float), marker="|", label="yolo")

    plt.xlim(x_min, x_max)              # <-- to last frame
    plt.ylim(0.0, 1.15)
    plt.xlabel("Frame")
    plt.ylabel("IoU")
    plt.title(_timeline_title("IoU"), fontsize=16)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower left", framealpha=0.9, ncol=2)
    plt.tight_layout()
    iou_png = str(out_prefix) + "_timeline_iou.png"
    plt.savefig(iou_png)
    plt.close()

    # ----- Center distance timeline -----
    plt.figure(figsize=(10.5, 4.6))
    if x_m.size:
        plt.plot(x_m, cd_m, label="Center dist (matched)")
    if np.isfinite(cd_mean):
        plt.axhline(cd_mean, color="tab:red", linestyle="--", linewidth=2.0,
                    label=f"Mean dist={cd_mean:.2f}px")

    # Place markers just above data (avoid squashing the curve)
    data_max = float(cd_m.max()) if cd_m.size else 0.0
    margin = max(1.0, 0.10 * max(1e-9, data_max))
    marker_y0 = data_max + margin
    marker_y1 = data_max + 2.0 * margin

    if reinit_f.size:
        plt.scatter(reinit_f, np.full_like(reinit_f, marker_y0, dtype=float), marker="|", label="reinit")
    if yolo_f.size:
        plt.scatter(yolo_f, np.full_like(yolo_f, marker_y1, dtype=float), marker="|", label="yolo")

    ymax = max(data_max + 3.0 * margin, marker_y1 + margin, 1.0)
    plt.xlim(x_min, x_max)              # <-- to last frame
    plt.ylim(0.0, ymax)
    plt.xlabel("Frame")
    plt.ylabel("Pixels")
    plt.title(_timeline_title("Center distance"), fontsize=16)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", framealpha=0.9, ncol=2)
    plt.tight_layout()
    cd_png = str(out_prefix) + "_timeline_center.png"
    plt.savefig(cd_png)
    plt.close()

    # ----------------------------
    # Structured summary
    # ----------------------------
    summary_txt = str(out_prefix) + "_summary.txt"

    IOU_THR_FIXED = 0.5
    PX_THR_FIXED = 20.0

    succ05_num = int((iou >= IOU_THR_FIXED).sum())
    prec20_num = int((cd <= PX_THR_FIXED).sum())

    denom_matched = int(matched_frames)
    denom_gt = int(gt_present)

    succ05_cond = float(succ05_num / denom_matched) if denom_matched else 0.0
    succ05_overall = float(succ05_num / denom_gt) if denom_gt else 0.0

    prec20_cond = float(prec20_num / denom_matched) if denom_matched else 0.0
    prec20_overall = float(prec20_num / denom_gt) if denom_gt else 0.0

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("OBJECT TRACKING EVALUATION SUMMARY\n")
        f.write("=================================\n\n")
        f.write(f"Source CSV: {args.csv}\n\n")

        f.write("1) Definitions\n")
        f.write("--------------\n")
        f.write("GT-present frame   : gt_has=1\n")
        f.write("GT-absent frame    : gt_has=0\n")
        f.write("Matched frame      : gt_has=1 AND pred_has=1\n\n")

        f.write("2) Dataset counts\n")
        f.write("-----------------\n")
        f.write(f"Total frames              : {total_frames}\n")
        f.write(f"GT-present frames         : {gt_present}\n")
        f.write(f"GT-absent frames          : {gt_absent}\n")
        f.write(f"Pred-present frames       : {pred_present}\n")
        f.write(f"Matched frames            : {matched_frames}\n")
        f.write(f"YOLO frames (yolo_has=1)  : {yolo_on}\n")
        f.write(f"Reinit events (reinit=1)  : {reinit_n}\n")
        f.write(f"Legend label              : {tracker_label}\n\n")

        f.write("3) Presence behavior\n")
        f.write("--------------------\n")
        f.write(f"Coverage (matched/GT-present) : {_fmt_ratio(matched_frames, gt_present)}\n")
        f.write(f"Miss rate (miss/GT-present)   : {_fmt_ratio(misses_gt_present, gt_present)}\n")
        f.write(f"FP rate (FP/GT-absent)        : {_fmt_ratio(fp_gt_absent, gt_absent)}\n\n")

        f.write("4) Point metrics\n")
        f.write("----------------\n")
        f.write("Success@IoU≥0.5\n")
        f.write(f"  Conditional: {succ05_num} / {denom_matched} = {succ05_cond:.3f}\n")
        f.write(f"  Overall    : {succ05_num} / {denom_gt} = {succ05_overall:.3f}\n\n")

        f.write("Precision@20px\n")
        f.write(f"  Conditional: {prec20_num} / {denom_matched} = {prec20_cond:.3f}\n")
        f.write(f"  Overall    : {prec20_num} / {denom_gt} = {prec20_overall:.3f}\n\n")

        f.write("5) Curve AUCs\n")
        f.write("-------------\n")
        f.write(f"AUC Conditional : {auc_cond:.3f}\n")
        f.write(f"AUC Overall     : {auc_overall:.3f}\n\n")

        f.write("6) Matched-only stats\n")
        f.write("---------------------\n")
        f.write(f"Mean IoU (matched)        : {_safe_mean(iou):.3f}\n")
        f.write(f"Mean center dist (matched): {_safe_mean(cd):.3f} px\n\n")

        f.write("7) Optional composite\n")
        f.write("---------------------\n")
        f.write(f"Effective AUC = AUC(Conditional) * Coverage : {effective_auc:.3f}\n\n")

        f.write("8) Output files\n")
        f.write("---------------\n")
        outputs = [
            otb_png,
            str(out_prefix) + "_precision_conditional.png",
            str(out_prefix) + "_precision_overall.png",
            str(out_prefix) + "_precision_both.png",
            str(out_prefix) + "_success_conditional.png",
            str(out_prefix) + "_success_overall.png",
            str(out_prefix) + "_success_both.png",
            iou_png,
            cd_png,
            summary_txt,
        ]
        for p in outputs:
            f.write(p + "\n")

    print("Saved summary:", summary_txt)
    print("Saved plots:", otb_png)
    print(
        f"Legend={tracker_label}  Coverage={coverage:.3f}  "
        f"P@{p_at:g}px(overall)={p_at_overall:.3f}  "
        f"AUC(overall)={auc_overall:.3f}  S@{thr:g}(overall)={succ_overall_at:.3f}"
    )


if __name__ == "__main__":
    main()
