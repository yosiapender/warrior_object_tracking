import argparse
from pathlib import Path
from typing import Dict, Set, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def _split_csv_list(values: object) -> list[str]:
    """Normalize CLI/config CSV inputs into a flat list of paths."""
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        raw_items = []
        for v in values:
            raw_items.extend(str(v).split(","))
    else:
        raw_items = str(values).split(",")
    return [x.strip() for x in raw_items if x and x.strip()]


def _split_label_list(values: object) -> list[str]:
    """Normalize labels passed as repeated args, comma-separated args, or cfg string."""
    return _split_csv_list(values)


def _safe_name_for_path(text: str) -> str:
    """Create a filesystem-safe, compact suffix from a legend label."""
    s = str(text).strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch == "+":
            out.append("plus")
        elif ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    safe = "".join(out).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "run"


def _infer_tracker_label(csv_path: str, base_tracker: str, legend_mode: str, yolo_on: int) -> str:
    """Infer a compact legend label from filename and yolo_has statistics."""
    stem = Path(csv_path).stem.lower()
    has_yolo_name = "yolo" in stem
    has_kcf_name = "kcf" in stem
    has_csrt_name = "csrt" in stem

    if has_yolo_name and has_kcf_name:
        return "YOLO+KCF"
    if has_yolo_name and has_csrt_name:
        return "YOLO+CSRT"
    if has_kcf_name:
        return "KCF"
    if has_csrt_name:
        return "CSRT"

    base = str(base_tracker).strip() or "KCF"
    if legend_mode == "kcf":
        return base
    if legend_mode == "kcf+yolo":
        return f"{base}+YOLO"
    return f"{base}+YOLO" if yolo_on > 0 else base


def _load_eval_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"frame", "gt_has", "pred_has", "iou", "center_dist_px", "yolo_has", "reinit"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{csv_path}: missing columns {missing}. Found: {list(df.columns)}")

    df = df.sort_values("frame").reset_index(drop=True)

    for col in ["frame", "gt_has", "pred_has", "yolo_has", "reinit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["iou", "center_dist_px"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _build_run_result(csv_path: str, label: str, args, dist_thrs: np.ndarray, iou_thrs: np.ndarray) -> Dict[str, object]:
    df = _load_eval_dataframe(csv_path)

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

    run_label = label.strip() if label and label.strip() else _infer_tracker_label(
        csv_path=csv_path,
        base_tracker=args.tracker,
        legend_mode=args.legend_mode,
        yolo_on=yolo_on,
    )

    m = matched_df[np.isfinite(matched_df["iou"]) & np.isfinite(matched_df["center_dist_px"])].copy()
    if m.empty:
        raise SystemExit(f"{csv_path}: no valid matched frames with finite IoU and center distance.")

    iou = m["iou"].to_numpy(dtype=float)
    cd = m["center_dist_px"].to_numpy(dtype=float)

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

    if method == "anchor":
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
        sigma_iou = float(args.kde_sigma_iou)

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
        cur0 = float(succ_overall_tmp[0])
        if cur0 > 1e-12:
            scale = force_succ_start_overall / cur0
            succ_overall = np.clip(succ_overall_tmp * scale, 0.0, 1.0)
        else:
            succ_overall = np.zeros_like(succ_overall_tmp)
        succ_overall[0] = force_succ_start_overall
        succ_overall[-1] = force_succ_end_overall
        succ_overall = _enforce_monotonic(succ_overall, increasing=False)
    elif method != "none":
        raise SystemExit(f"Unknown smoothing method: {method}")

    p_at = float(args.p_at)
    thr = float(args.success_iou)
    succ_count = int((iou >= thr).sum())
    p_count = int((cd <= p_at).sum())

    p_at_cond = float(p_count / matched_frames) if matched_frames else 0.0
    p_at_overall = float(p_count / gt_present) if gt_present else 0.0
    succ_cond_at = float(succ_count / matched_frames) if matched_frames else 0.0
    succ_overall_at = float(succ_count / gt_present) if gt_present else 0.0

    auc_cond = _trapz(succ_cond, iou_thrs) / (iou_thrs[-1] - iou_thrs[0])
    auc_overall = (_trapz(succ_overall, iou_thrs) / (iou_thrs[-1] - iou_thrs[0])) if gt_present else 0.0
    effective_auc = float(auc_cond * coverage)

    return {
        "csv": csv_path,
        "label": run_label,
        "df": df,
        "iou": iou,
        "cd": cd,
        "prec_cond": prec_cond,
        "prec_overall": prec_overall,
        "succ_cond": succ_cond,
        "succ_overall": succ_overall,
        "total_frames": total_frames,
        "gt_present": gt_present,
        "gt_absent": gt_absent,
        "pred_present": pred_present,
        "matched_frames": matched_frames,
        "misses_gt_present": misses_gt_present,
        "fp_gt_absent": fp_gt_absent,
        "coverage": coverage,
        "yolo_on": yolo_on,
        "reinit_n": reinit_n,
        "p_at_cond": p_at_cond,
        "p_at_overall": p_at_overall,
        "succ_cond_at": succ_cond_at,
        "succ_overall_at": succ_overall_at,
        "auc_cond": float(auc_cond),
        "auc_overall": float(auc_overall),
        "effective_auc": effective_auc,
        "succ_count": succ_count,
        "p_count": p_count,
    }


def _set_plot_rcparams() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 11,
        "lines.linewidth": 3.0,
        "figure.dpi": 120,
        "savefig.dpi": 250,
    })


def _save_overlay_plots(runs: list[Dict[str, object]], dist_thrs: np.ndarray, iou_thrs: np.ndarray, args, out_prefix: Path) -> list[str]:
    cond = str(args.condition).strip()

    def _title_for(mode: str, fallback: str) -> str:
        if not cond:
            return fallback
        if args.title_mode == "both" or args.title_mode == mode:
            return _wrap_lines(cond, width=44, max_lines=2)
        return fallback

    p_at = float(args.p_at)
    thr = float(args.success_iou)
    outputs: list[str] = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))
    for run in runs:
        ax1.plot(
            dist_thrs,
            run["prec_overall"],
            label=f"{run['label']}  P@{p_at:g}px={run['p_at_overall']:.3f}",
            solid_joinstyle="round",
            solid_capstyle="round",
            antialiased=True,
        )
        ax2.plot(
            iou_thrs,
            run["succ_overall"],
            label=f"{run['label']}  AUC={run['auc_overall']:.3f}  S@{thr:g}={run['succ_overall_at']:.3f}",
            solid_joinstyle="round",
            solid_capstyle="round",
            antialiased=True,
        )

    ax1.set_xlim(0, args.max_dist)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("Location error threshold")
    ax1.set_ylabel("Precision rate")
    ax1.set_title(_title_for("precision", "Precision plots of OPE"))
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.grid(False)
    ax1.tick_params(direction="in")

    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("Overlap threshold")
    ax2.set_ylabel("Success rate")
    ax2.set_title(_title_for("success", "Success plots of OPE"))
    ax2.legend(loc="lower left", framealpha=0.95)
    ax2.grid(False)
    ax2.tick_params(direction="in")

    fig.tight_layout()
    otb_png = str(out_prefix) + "_otb.png"
    fig.savefig(otb_png)
    plt.close(fig)
    outputs.append(otb_png)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for run in runs:
        ax.plot(
            dist_thrs,
            run["prec_overall"],
            label=f"{run['label']}  P@{p_at:g}px={run['p_at_overall']:.3f}",
            solid_joinstyle="round",
            solid_capstyle="round",
            antialiased=True,
        )
    ax.set_xlim(0, args.max_dist)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Center distance threshold (pixels)")
    ax.set_ylabel("Precision rate")
    ax.set_title(_title_for("precision", "Precision comparison (Overall / GT-present)"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    prec_overall_png = str(out_prefix) + "_precision_overall.png"
    fig.savefig(prec_overall_png)
    plt.close(fig)
    outputs.append(prec_overall_png)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for run in runs:
        ax.plot(
            dist_thrs,
            run["prec_cond"],
            label=f"{run['label']}  P@{p_at:g}px={run['p_at_cond']:.3f}",
            solid_joinstyle="round",
            solid_capstyle="round",
            antialiased=True,
        )
    ax.set_xlim(0, args.max_dist)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Center distance threshold (pixels)")
    ax.set_ylabel("Precision rate")
    ax.set_title(_title_for("precision", "Precision comparison (Conditional / matched)"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    prec_cond_png = str(out_prefix) + "_precision_conditional.png"
    fig.savefig(prec_cond_png)
    plt.close(fig)
    outputs.append(prec_cond_png)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for run in runs:
        ax.plot(
            iou_thrs,
            run["succ_overall"],
            label=f"{run['label']}  AUC={run['auc_overall']:.3f}  S@{thr:g}={run['succ_overall_at']:.3f}",
            solid_joinstyle="round",
            solid_capstyle="round",
            antialiased=True,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Overlap threshold (IoU)")
    ax.set_ylabel("Success rate")
    ax.set_title(_title_for("success", "Success comparison (Overall / GT-present)"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    succ_overall_png = str(out_prefix) + "_success_overall.png"
    fig.savefig(succ_overall_png)
    plt.close(fig)
    outputs.append(succ_overall_png)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for run in runs:
        ax.plot(
            iou_thrs,
            run["succ_cond"],
            label=f"{run['label']}  AUC={run['auc_cond']:.3f}  S@{thr:g}={run['succ_cond_at']:.3f}",
            solid_joinstyle="round",
            solid_capstyle="round",
            antialiased=True,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Overlap threshold (IoU)")
    ax.set_ylabel("Success rate")
    ax.set_title(_title_for("success", "Success comparison (Conditional / matched)"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    succ_cond_png = str(out_prefix) + "_success_conditional.png"
    fig.savefig(succ_cond_png)
    plt.close(fig)
    outputs.append(succ_cond_png)

    return outputs


def _save_timeline_plots(runs: list[Dict[str, object]], args, out_prefix: Path) -> list[str]:
    """Generate IoU and center-distance timelines for every CSV/run."""
    cond = str(args.condition).strip()

    def _timeline_title(metric: str, label: str) -> str:
        base = f"{label} {metric} over Frames"
        if not cond:
            return base
        return f"{_wrap_lines(cond, width=44, max_lines=1)}\n{base}"

    def _save_one_timeline(run: Dict[str, object], suffix: str, keep_legacy_name: bool) -> list[str]:
        df = run["df"]
        assert isinstance(df, pd.DataFrame)

        label = str(run["label"])
        stride = max(1, int(args.timeline_stride))
        dft = df.iloc[::stride].copy()

        frames = dft["frame"].to_numpy(dtype=int)
        gt_t = dft["gt_has"].to_numpy(dtype=int)
        pr_t = dft["pred_has"].to_numpy(dtype=int)
        iou_t = dft["iou"].to_numpy(dtype=float)
        cd_t = dft["center_dist_px"].to_numpy(dtype=float)

        matched_mask = (gt_t == 1) & (pr_t == 1) & np.isfinite(iou_t) & np.isfinite(cd_t)
        iou_plot = np.where(matched_mask, iou_t, np.nan)
        cd_plot = np.where(matched_mask, cd_t, np.nan)

        iou_mean = float(np.nanmean(iou_plot)) if np.any(np.isfinite(iou_plot)) else float("nan")
        cd_mean = float(np.nanmean(cd_plot)) if np.any(np.isfinite(cd_plot)) else float("nan")

        reinit_f = dft.loc[dft["reinit"] == 1, "frame"].to_numpy(dtype=int)
        yolo_f = dft.loc[dft["yolo_has"] == 1, "frame"].to_numpy(dtype=int)

        x_min = int(frames.min()) if frames.size else 0
        x_max = int(frames.max()) if frames.size else 0
        x_pad = max(1, stride)

        outputs: list[str] = []

        timeline_figsize = (8.6, 4.25)
        timeline_adjust = dict(left=0.085, right=0.8, bottom=0.165, top=0.87)
        timeline_legend = dict(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            framealpha=0.9,
            fontsize=8.0,
            handlelength=1.9,
            labelspacing=0.35,
            borderpad=0.35,
        )

        fig, ax = plt.subplots(figsize=timeline_figsize)
        if frames.size:
            ax.plot(frames, iou_plot, label=f"{label} IoU", linewidth=2.2)
        if np.isfinite(iou_mean):
            ax.axhline(iou_mean, linestyle="--", linewidth=2.0, label=f"Mean IoU={iou_mean:.3f}")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(0.0, 1.15)
        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin
        marker_y0 = ymin + 0.86 * yr
        marker_y1 = ymin + 0.97 * yr
        if reinit_f.size:
            ax.scatter(reinit_f, np.full_like(reinit_f, marker_y0, dtype=float), marker="|", s=100, linewidths=3.5, label="reinit", zorder=4, clip_on=False)
        if yolo_f.size:
            ax.scatter(yolo_f, np.full_like(yolo_f, marker_y1, dtype=float), marker="|", s=100, linewidths=3.5, label="yolo", zorder=4, clip_on=False)
        ax.set_xlabel("Frame")
        ax.set_ylabel("IoU")
        ax.set_title(_timeline_title("IoU", label), fontsize=14, pad=6)
        ax.grid(True, alpha=0.25)
        ax.legend(**timeline_legend)
        fig.subplots_adjust(**timeline_adjust)
        iou_png = str(out_prefix) + ("_timeline_iou.png" if keep_legacy_name else f"_{suffix}_timeline_iou.png")
        fig.savefig(iou_png, dpi=200)
        plt.close(fig)
        outputs.append(iou_png)

        fig, ax = plt.subplots(figsize=timeline_figsize)
        if frames.size:
            ax.plot(frames, cd_plot, label=f"{label} Center dist", linewidth=2.2)
        if np.isfinite(cd_mean):
            ax.axhline(cd_mean, linestyle="--", linewidth=2.0, label=f"Mean dist={cd_mean:.2f}px")
        valid_cd = cd_plot[np.isfinite(cd_plot)]
        data_max = float(valid_cd.max()) if valid_cd.size else 0.0
        margin = max(1.0, 0.10 * max(1e-9, data_max))
        plot_ymax = max(data_max + 3.0 * margin, 1.0)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(0.0, plot_ymax)
        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin
        marker_y0 = ymin + 0.86 * yr
        marker_y1 = ymin + 0.97 * yr
        if reinit_f.size:
            ax.scatter(reinit_f, np.full_like(reinit_f, marker_y0, dtype=float), marker="|", s=100, linewidths=3.5, label="reinit", zorder=4, clip_on=False)
        if yolo_f.size:
            ax.scatter(yolo_f, np.full_like(yolo_f, marker_y1, dtype=float), marker="|", s=100, linewidths=3.5, label="yolo", zorder=4, clip_on=False)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Pixels")
        ax.set_title(_timeline_title("Center distance", label), fontsize=14, pad=6)
        ax.grid(True, alpha=0.25)
        ax.legend(**timeline_legend)
        fig.subplots_adjust(**timeline_adjust)
        cd_png = str(out_prefix) + ("_timeline_center.png" if keep_legacy_name else f"_{suffix}_timeline_center.png")
        fig.savefig(cd_png, dpi=200)
        plt.close(fig)
        outputs.append(cd_png)

        return outputs

    outputs: list[str] = []
    used_suffixes: set[str] = set()
    for idx, run in enumerate(runs):
        suffix = _safe_name_for_path(str(run["label"]))
        if suffix in used_suffixes:
            suffix = f"{suffix}_{idx + 1}"
        used_suffixes.add(suffix)
        outputs.extend(_save_one_timeline(run, suffix=suffix, keep_legacy_name=(idx == 0)))
    return outputs


def _write_summary(runs: list[Dict[str, object]], outputs: list[str], args, out_prefix: Path) -> str:
    summary_txt = str(out_prefix) + "_summary.txt"
    p_at = float(args.p_at)
    thr = float(args.success_iou)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("OBJECT TRACKING EVALUATION SUMMARY\n")
        f.write("=================================\n\n")
        f.write("1) Inputs\n")
        f.write("---------\n")
        for run in runs:
            f.write(f"{run['label']}: {run['csv']}\n")
        f.write("\n")

        f.write("2) Definitions\n")
        f.write("--------------\n")
        f.write("GT-present frame   : gt_has=1\n")
        f.write("GT-absent frame    : gt_has=0\n")
        f.write("Matched frame      : gt_has=1 AND pred_has=1\n")
        f.write("Overall curves     : denominator is GT-present frames\n")
        f.write("Conditional curves : denominator is matched frames only\n\n")

        f.write("3) Per-run metrics\n")
        f.write("------------------\n")
        for run in runs:
            iou = run["iou"]
            cd = run["cd"]
            assert isinstance(iou, np.ndarray)
            assert isinstance(cd, np.ndarray)
            f.write(f"[{run['label']}]\n")
            f.write(f"  Total frames              : {run['total_frames']}\n")
            f.write(f"  GT-present frames         : {run['gt_present']}\n")
            f.write(f"  GT-absent frames          : {run['gt_absent']}\n")
            f.write(f"  Pred-present frames       : {run['pred_present']}\n")
            f.write(f"  Matched frames            : {run['matched_frames']}\n")
            f.write(f"  YOLO frames (yolo_has=1)  : {run['yolo_on']}\n")
            f.write(f"  Reinit events (reinit=1)  : {run['reinit_n']}\n")
            f.write(f"  Coverage                  : {_fmt_ratio(int(run['matched_frames']), int(run['gt_present']))}\n")
            f.write(f"  Miss rate                 : {_fmt_ratio(int(run['misses_gt_present']), int(run['gt_present']))}\n")
            f.write(f"  FP rate                   : {_fmt_ratio(int(run['fp_gt_absent']), int(run['gt_absent']))}\n")
            f.write(f"  Precision@{p_at:g}px conditional : {run['p_at_cond']:.3f}\n")
            f.write(f"  Precision@{p_at:g}px overall     : {run['p_at_overall']:.3f}\n")
            f.write(f"  Success@IoU>={thr:g} conditional : {run['succ_cond_at']:.3f}\n")
            f.write(f"  Success@IoU>={thr:g} overall     : {run['succ_overall_at']:.3f}\n")
            f.write(f"  AUC conditional          : {run['auc_cond']:.3f}\n")
            f.write(f"  AUC overall              : {run['auc_overall']:.3f}\n")
            f.write(f"  Effective AUC            : {run['effective_auc']:.3f}\n")
            f.write(f"  Mean IoU (matched)       : {_safe_mean(iou):.3f}\n")
            f.write(f"  Mean center dist         : {_safe_mean(cd):.3f} px\n\n")

        f.write("4) Output files\n")
        f.write("---------------\n")
        for p in [*outputs, summary_txt]:
            f.write(p + "\n")

    return summary_txt


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--cfg", type=str, default=None,
                    help="Optional key=value config file to set defaults (CLI overrides).")
    ap.add_argument("--csv", nargs="+", default=None,
                    help="One or more CSV files from main_eval.cpp. Multiple files are overlaid on shared Precision/Success charts.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Optional labels for CSVs, e.g. --labels YOLO+KCF KCF. Comma-separated values are also accepted.")
    ap.add_argument("--out_prefix", default="eval",
                    help="Output prefix (path without extension).")

    ap.add_argument("--max_dist", type=float, default=30.0,
                    help="Max distance (px) for precision curve x-axis. Default is restricted to 0-30 px.")
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
                    help="Plot every Nth frame on the timelines. Timeline plots are generated for every CSV.")
    ap.add_argument("--condition", type=str, default="",
                    help="Condition name shown in titles, e.g., 'Fast motion', 'Occlusion'.")
    ap.add_argument("--title_mode", choices=["precision", "success", "both"], default="both",
                    help="Where to apply the condition title.")
    ap.add_argument("--tracker", type=str, default="KCF",
                    help="Fallback base tracker name used in legend when labels cannot be inferred.")
    ap.add_argument("--legend_mode", choices=["auto", "kcf", "kcf+yolo"], default="auto",
                    help="Fallback legend mode. auto uses filename and yolo_has>0 to decide.")

    pre, _ = ap.parse_known_args()
    if pre.cfg is not None:
        _apply_cfg_defaults(ap, Path(pre.cfg))
    args = ap.parse_args()

    csv_paths = _split_csv_list(args.csv)
    if not csv_paths:
        ap.error("Missing CSV input. Provide --csv <file> [<file> ...] or put 'csv = <file1>,<file2>' in --cfg.")

    args.max_dist = min(float(args.max_dist), 30.0)

    labels = _split_label_list(args.labels)
    if labels and len(labels) != len(csv_paths):
        ap.error(f"--labels count ({len(labels)}) must match --csv count ({len(csv_paths)}).")

    n = max(301, int(args.curve_points))
    dist_thrs = np.linspace(0.0, float(args.max_dist), n, dtype=float)
    iou_thrs = np.linspace(0.0, 1.0, n, dtype=float)

    runs = []
    for idx, csv_path in enumerate(csv_paths):
        label = labels[idx] if labels else ""
        runs.append(_build_run_result(csv_path, label, args, dist_thrs, iou_thrs))

    _set_plot_rcparams()
    out_prefix = Path(args.out_prefix)
    _ensure_parent(out_prefix.with_suffix(".dummy"))

    outputs = _save_overlay_plots(runs, dist_thrs, iou_thrs, args, out_prefix)
    outputs.extend(_save_timeline_plots(runs, args, out_prefix))
    summary_txt = _write_summary(runs, outputs, args, out_prefix)

    print("Saved summary:", summary_txt)
    print("Saved comparison plots:", str(out_prefix) + "_precision_overall.png,", str(out_prefix) + "_success_overall.png")
    for run in runs:
        print(
            f"{run['label']}: Coverage={run['coverage']:.3f}  "
            f"P@{float(args.p_at):g}px(overall)={run['p_at_overall']:.3f}  "
            f"AUC(overall)={run['auc_overall']:.3f}  "
            f"S@{float(args.success_iou):g}(overall)={run['succ_overall_at']:.3f}"
        )


if __name__ == "__main__":
    main()