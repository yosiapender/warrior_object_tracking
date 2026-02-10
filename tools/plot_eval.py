import argparse
from pathlib import Path

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
    # NumPy 2.x removed np.trapz; np.trapezoid is the replacement.
    trap = getattr(np, "trapezoid", None)
    if trap is None:
        trap = np.trapz
    return float(trap(y, x))


# ----------------------------
# Core computation helpers
# ----------------------------
def _compute_curves(iou: np.ndarray,
                    cd: np.ndarray,
                    gt_present: int,
                    matched_frames: int,
                    dist_thrs: np.ndarray,
                    iou_thrs: np.ndarray):
    """
    Returns:
      precision_cond(d)  = P(cd <= d | matched)
      precision_overall(d) = P(cd <= d over GT-present frames) = #(cd<=d)/#(GT-present)

      success_cond(t)    = P(iou >= t | matched)
      success_overall(t) = #(iou>=t)/#(GT-present)
    """
    # Conditional (matched-only)
    precision_cond = np.array([(cd <= d).mean() for d in dist_thrs], dtype=float)
    success_cond = np.array([(iou >= t).mean() for t in iou_thrs], dtype=float)

    # Overall on GT-present denominator (misses reduce score automatically)
    if gt_present > 0:
        precision_overall = np.array([((cd <= d).sum() / gt_present) for d in dist_thrs], dtype=float)
        success_overall = np.array([((iou >= t).sum() / gt_present) for t in iou_thrs], dtype=float)
    else:
        precision_overall = np.zeros_like(dist_thrs, dtype=float)
        success_overall = np.zeros_like(iou_thrs, dtype=float)

    return precision_cond, precision_overall, success_cond, success_overall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="CSV from main_eval.cpp: frame, gt_has, pred_has, iou, center_dist_px, yolo_has, reinit")
    ap.add_argument("--out_prefix", default="eval",
                    help="Output prefix (e.g., data/output/occlusion)")
    ap.add_argument("--max_dist", type=float, default=50.0,
                    help="Max distance threshold for precision curves")
    ap.add_argument("--dist_step", type=float, default=1.0,
                    help="Distance step for precision curve")
    ap.add_argument("--iou_step", type=float, default=0.01,
                    help="IoU step for success curve")
    ap.add_argument("--p_at", type=float, default=20.0,
                    help="Precision point metric threshold (pixels), default OTB=20")
    ap.add_argument("--success_iou", type=float, default=0.5,
                    help="Success point metric IoU threshold, default=0.5")
    ap.add_argument("--timeline_stride", type=int, default=1,
                    help="Plot every Nth frame for timeline plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = {"frame", "gt_has", "pred_has", "iou", "center_dist_px", "yolo_has", "reinit"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns {missing}. Found: {list(df.columns)}")

    df = df.sort_values("frame").reset_index(drop=True)

    # Coerce types robustly
    for col in ["frame", "gt_has", "pred_has", "yolo_has", "reinit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["iou", "center_dist_px"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    total_frames = int(len(df))
    gt_present = int((df["gt_has"] == 1).sum())
    gt_absent = total_frames - gt_present

    pred_present = int((df["pred_has"] == 1).sum())
    matched_df = df[(df["gt_has"] == 1) & (df["pred_has"] == 1)].copy()
    matched_frames = int(len(matched_df))

    misses_gt_present = int(((df["gt_has"] == 1) & (df["pred_has"] == 0)).sum())
    fp_gt_absent = int(((df["gt_has"] == 0) & (df["pred_has"] == 1)).sum())

    coverage = (matched_frames / gt_present) if gt_present else 0.0
    miss_rate = (misses_gt_present / gt_present) if gt_present else 0.0
    fp_rate_absent = (fp_gt_absent / gt_absent) if gt_absent else 0.0

    yolo_on = int((df["yolo_has"] == 1).sum())
    reinit_n = int((df["reinit"] == 1).sum())

    # Valid matched metrics only
    m = matched_df[np.isfinite(matched_df["iou"]) & np.isfinite(matched_df["center_dist_px"])].copy()
    if m.empty:
        raise SystemExit("No valid matched frames with finite IoU and center distance.")

    iou = m["iou"].to_numpy(dtype=float)
    cd = m["center_dist_px"].to_numpy(dtype=float)

    # Threshold grids
    dist_thrs = np.arange(0.0, args.max_dist + 1e-9, args.dist_step, dtype=float)
    iou_thrs = np.arange(0.0, 1.0 + 1e-9, args.iou_step, dtype=float)

    # Curves (two paradigms)
    prec_cond, prec_overall, succ_cond, succ_overall = _compute_curves(
        iou=iou,
        cd=cd,
        gt_present=gt_present,
        matched_frames=matched_frames,
        dist_thrs=dist_thrs,
        iou_thrs=iou_thrs,
    )

    # Point metrics
    p_at = float(args.p_at)
    thr = float(args.success_iou)

    # Conditional (matched-only) point metrics
    p_at_cond = float((cd <= p_at).mean())
    succ_count = int((iou >= thr).sum())
    succ_cond_at = float((succ_count / matched_frames) if matched_frames else 0.0)

    # Overall (GT-present) point metrics
    p_at_overall = float(((cd <= p_at).sum() / gt_present) if gt_present else 0.0)
    succ_overall_at = float((succ_count / gt_present) if gt_present else 0.0)

    # AUCs
    auc_cond = _trapz(succ_cond, iou_thrs) / (iou_thrs[-1] - iou_thrs[0])
    auc_overall = (_trapz(succ_overall, iou_thrs) / (iou_thrs[-1] - iou_thrs[0])) if gt_present else 0.0

    # Optional composite: penalize misses while keeping interpretability
    effective_auc = float(auc_cond * coverage)

    # ----------------------------
    # Plot styling
    # ----------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "lines.linewidth": 2.5,
        "figure.dpi": 120,
        "savefig.dpi": 250,
    })

    out_prefix = Path(args.out_prefix)
    _ensure_parent(out_prefix.with_suffix(".dummy"))

    # ----------------------------
    # Precision plots: separate + combined
    # ----------------------------
    # Separate: Conditional (Matched-only)
    plt.figure(figsize=(7.8, 5.4))
    plt.plot(dist_thrs, prec_cond,
             label=f"Conditional: P@{p_at:g}px = {p_at_cond:.3f}  (denom = matched)")
    plt.xlim(0, args.max_dist)
    plt.ylim(0, 1.0)
    plt.xlabel("Location error threshold (pixels)")
    plt.ylabel("Precision rate")
    plt.title("Precesion curve(Conditional / matched)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    prec_cond_png = str(out_prefix) + "_precision_conditional.png"
    plt.savefig(prec_cond_png)
    plt.close()

    # Separate: Overall (GT-present)
    plt.figure(figsize=(7.8, 5.4))
    plt.plot(dist_thrs, prec_overall,
             label=f"Overall: P@{p_at:g}px = {p_at_overall:.3f}  (denom = GT-present)")
    plt.xlim(0, args.max_dist)
    plt.ylim(0, 1.0)
    plt.xlabel("Location error threshold (pixels)")
    plt.ylabel("Precision rate")
    plt.title("Precesion curve(Overall / GT-present)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    prec_overall_png = str(out_prefix) + "_precision_overall.png"
    plt.savefig(prec_overall_png)
    plt.close()

    # Combined
    plt.figure(figsize=(7.8, 5.4))
    plt.plot(dist_thrs, prec_cond,
             label=f"Conditional (matched): P@{p_at:g}px={p_at_cond:.3f}")
    plt.plot(dist_thrs, prec_overall,
             label=f"Overall (GT-present): P@{p_at:g}px={p_at_overall:.3f}  (Coverage={coverage:.3f})")
    plt.xlim(0, args.max_dist)
    plt.ylim(0, 1.0)
    plt.xlabel("Location error threshold (pixels)")
    plt.ylabel("Precision rate")
    plt.title("Precision curves — Conditional vs Overall")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    prec_both_png = str(out_prefix) + "_precision_both.png"
    plt.savefig(prec_both_png)
    plt.close()

    # ----------------------------
    # Success plots: separate + combined
    # ----------------------------
    # Separate: Conditional (Matched-only)
    plt.figure(figsize=(7.8, 5.4))
    plt.plot(iou_thrs, succ_cond,
             label=f"Conditional: AUC={auc_cond:.3f}, Success@{thr:g}={succ_cond_at:.3f} (denom=matched)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Overlap threshold (IoU)")
    plt.ylabel("Success rate")
    plt.title("Success curve(Conditional / matched)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    succ_cond_png = str(out_prefix) + "_success_conditional.png"
    plt.savefig(succ_cond_png)
    plt.close()

    # Separate: Overall (GT-present)
    plt.figure(figsize=(7.8, 5.4))
    plt.plot(iou_thrs, succ_overall,
             label=f"Overall: AUC={auc_overall:.3f}, Success@{thr:g}={succ_overall_at:.3f} (denom=GT-present)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Overlap threshold (IoU)")
    plt.ylabel("Success rate")
    plt.title("Success curve(Overall / GT-present)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    succ_overall_png = str(out_prefix) + "_success_overall.png"
    plt.savefig(succ_overall_png)
    plt.close()

    # Combined
    plt.figure(figsize=(7.8, 5.4))
    plt.plot(iou_thrs, succ_cond, label=f"Conditional (matched): AUC={auc_cond:.3f}")
    plt.plot(iou_thrs, succ_overall, label=f"Overall (GT-present): AUC={auc_overall:.3f} (Coverage={coverage:.3f})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Overlap threshold (IoU)")
    plt.ylabel("Success rate")
    plt.title("Success curves — Conditional vs Overall")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    succ_both_png = str(out_prefix) + "_success_both.png"
    plt.savefig(succ_both_png)
    plt.close()

    # ----------------------------
    # Timeline plots (debug)
    # ----------------------------
    stride = max(1, int(args.timeline_stride))
    dft = df.iloc[::stride].copy()

    frames = dft["frame"].to_numpy(dtype=int)
    iou_t = dft["iou"].to_numpy(dtype=float)
    cd_t = dft["center_dist_px"].to_numpy(dtype=float)

    matched_mask = (dft["gt_has"].to_numpy(dtype=int) == 1) & (dft["pred_has"].to_numpy(dtype=int) == 1)
    iou_plot = np.where(matched_mask & np.isfinite(iou_t), iou_t, np.nan)
    cd_plot = np.where(matched_mask & np.isfinite(cd_t), cd_t, np.nan)

    reinit_f = dft.loc[dft["reinit"] == 1, "frame"].to_numpy(dtype=int)
    yolo_f = dft.loc[dft["yolo_has"] == 1, "frame"].to_numpy(dtype=int)

    plt.figure(figsize=(10.5, 4.6))
    plt.plot(frames, iou_plot, label="IoU (matched frames only)")
    if reinit_f.size:
        plt.scatter(reinit_f, np.full_like(reinit_f, 1.02, dtype=float), marker="|", label="reinit")
    if yolo_f.size:
        plt.scatter(yolo_f, np.full_like(yolo_f, 1.08, dtype=float), marker="|", label="yolo_has")
    plt.ylim(0.0, 1.15)
    plt.xlabel("Frame")
    plt.ylabel("IoU")
    plt.title("IoU timeline (gaps = no GT or no prediction)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", framealpha=0.9, ncol=3)
    plt.tight_layout()
    iou_png = str(out_prefix) + "_timeline_iou.png"
    plt.savefig(iou_png)
    plt.close()

    plt.figure(figsize=(10.5, 4.6))
    plt.plot(frames, cd_plot, label="Center distance (matched frames only)")
    if reinit_f.size:
        plt.scatter(reinit_f, np.full_like(reinit_f, args.max_dist * 1.02, dtype=float), marker="|", label="reinit")
    if yolo_f.size:
        plt.scatter(yolo_f, np.full_like(yolo_f, args.max_dist * 1.08, dtype=float), marker="|", label="yolo_has")
    ymax = args.max_dist
    if np.isfinite(cd_plot).any():
        ymax = max(ymax, float(np.nanmax(cd_plot)))
    plt.ylim(0.0, ymax * 1.15)
    plt.xlabel("Frame")
    plt.ylabel("Pixels")
    plt.title("Center distance timeline (gaps = no GT or no prediction)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", framealpha=0.9, ncol=3)
    plt.tight_layout()
    cd_png = str(out_prefix) + "_timeline_center.png"
    plt.savefig(cd_png)
    plt.close()

    # ----------------------------
    # Structured summary
    # ----------------------------
    summary_txt = str(out_prefix) + "_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("OBJECT TRACKING EVALUATION SUMMARY\n")
        f.write("=================================\n\n")
        f.write(f"Source CSV: {args.csv}\n\n")

        f.write("1) Definitions (to avoid confusion)\n")
        f.write("----------------------------------\n")
        f.write("GT-present frame   : gt_has=1 (object visible; must track)\n")
        f.write("GT-absent frame    : gt_has=0 (object not visible; should NOT output box)\n")
        f.write("Matched frame      : gt_has=1 AND pred_has=1 (IoU/center error are defined)\n\n")
        f.write("Conditional metrics (Matched-only): computed over matched frames only.\n")
        f.write("  Interpretation: 'When we output a box, how accurate is it?'\n")
        f.write("Overall metrics (GT-present denom): computed over all GT-present frames.\n")
        f.write("  Interpretation: 'Across all visible-object frames, how often are we correct?'\n")
        f.write("  Note: misses (pred_has=0 on gt_has=1) reduce overall metrics automatically.\n\n")

        f.write("2) Dataset counts\n")
        f.write("-----------------\n")
        f.write(f"Total frames             : {total_frames}\n")
        f.write(f"GT-present frames        : {gt_present}\n")
        f.write(f"GT-absent frames         : {gt_absent}\n")
        f.write(f"Pred-present frames      : {pred_present}\n")
        f.write(f"Matched frames           : {matched_frames}\n")
        f.write(f"YOLO frames (yolo_has=1)  : {yolo_on}\n")
        f.write(f"Reinit events (reinit=1)  : {reinit_n}\n\n")

        f.write("3) Robustness / Presence behavior (objective)\n")
        f.write("--------------------------------------------\n")
        f.write(f"Coverage on GT-present (matched / GT-present) : {matched_frames} / {gt_present} = {coverage:.3f}\n")
        f.write(f"Miss rate on GT-present (misses / GT-present) : {misses_gt_present} / {gt_present} = {miss_rate:.3f}\n")
        f.write(f"False positives on GT-absent (FP / GT-absent) : {fp_gt_absent} / {gt_absent} = {fp_rate_absent:.3f}\n\n")

        f.write("4) Success / Precision point metrics\n")
        f.write("-----------------------------------\n")
        f.write(f"Success@IoU≥{thr:g}\n")
        f.write(f"  Conditional (matched denom): {succ_count} / {matched_frames} = {succ_cond_at:.3f}\n")
        f.write(f"  Overall (GT-present denom) : {succ_count} / {gt_present} = {succ_overall_at:.3f}\n\n")
        f.write(f"Precision@{p_at:g}px (center error ≤ {p_at:g})\n")
        f.write(f"  Conditional (matched denom): {(cd <= p_at).sum()} / {matched_frames} = {p_at_cond:.3f}\n")
        f.write(f"  Overall (GT-present denom) : {(cd <= p_at).sum()} / {gt_present} = {p_at_overall:.3f}\n\n")

        f.write("5) Curve AUCs (Success curves)\n")
        f.write("------------------------------\n")
        f.write(f"AUC Conditional (matched-only) : {auc_cond:.3f}\n")
        f.write(f"AUC Overall (GT-present denom) : {auc_overall:.3f}\n\n")

        f.write("6) Matched-only quality statistics\n")
        f.write("---------------------------------\n")
        f.write(f"Mean IoU (matched)        : {_safe_mean(iou):.3f}\n")
        f.write(f"Mean center dist (matched): {_safe_mean(cd):.3f} px\n\n")

        f.write("7) Optional composite\n")
        f.write("---------------------\n")
        f.write(f"Effective AUC = AUC(Conditional) * Coverage : {effective_auc:.3f}\n\n")

        f.write("8) Output files\n")
        f.write("---------------\n")
        outputs = [
            prec_cond_png, prec_overall_png, prec_both_png,
            succ_cond_png, succ_overall_png, succ_both_png,
            iou_png, cd_png,
            summary_txt
        ]
        for p in outputs:
            f.write(p + "\n")

    # Console digest (short)
    print("Saved summary:", summary_txt)
    print("Saved plots:")
    for p in [
        prec_cond_png, prec_overall_png, prec_both_png,
        succ_cond_png, succ_overall_png, succ_both_png,
        iou_png, cd_png
    ]:
        print(" ", p)

    print("\nQuick digest:")
    print(f"  Coverage (GT-present): {coverage:.3f}")
    print(f"  Success@{thr:g} Conditional (matched): {succ_cond_at:.3f}")
    print(f"  Success@{thr:g} Overall (GT-present): {succ_overall_at:.3f}")
    print(f"  AUC Conditional: {auc_cond:.3f}")
    print(f"  AUC Overall:     {auc_overall:.3f}")
    print(f"  FP rate (GT-absent): {fp_rate_absent:.3f}")


if __name__ == "__main__":
    main()
