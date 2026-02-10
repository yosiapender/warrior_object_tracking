import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="CSV must contain: gt_has, pred_has, iou, center_dist_px")
    ap.add_argument("--out_prefix", default="eval_matched",
                    help="Output prefix for PNG files")
    ap.add_argument("--max_dist", type=float, default=50.0,
                    help="Max distance threshold for precision plot")
    ap.add_argument("--dist_step", type=float, default=1.0,
                    help="Distance step")
    ap.add_argument("--iou_step", type=float, default=0.01,
                    help="IoU step")
    ap.add_argument("--p_at", type=float, default=20.0,
                    help="Report precision at this threshold (OTB = 20)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = {"gt_has", "pred_has", "iou", "center_dist_px"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns {missing}. Found: {list(df.columns)}")

    # MATCHED-ONLY: keep frames where GT exists AND prediction exists
    m = df[(df["gt_has"] == 1) & (df["pred_has"] == 1)].copy()
    n = len(m)
    if n == 0:
        raise SystemExit("No matched frames (gt_has==1 & pred_has==1).")

    iou = m["iou"].to_numpy(dtype=float)
    cd  = m["center_dist_px"].to_numpy(dtype=float)

    # If any invalid remains, drop them (should not happen if CSV is clean)
    good = np.isfinite(iou) & np.isfinite(cd) & (iou >= 0.0) & (cd >= 0.0)
    iou = iou[good]
    cd  = cd[good]
    n2 = len(cd)
    if n2 == 0:
        raise SystemExit("Matched frames exist, but all have invalid iou/center_dist.")

    # Precision curve
    dist_thrs = np.arange(0.0, args.max_dist + 1e-9, args.dist_step, dtype=float)
    precision = np.array([(cd <= t).mean() for t in dist_thrs], dtype=float)
    p_at_val = float((cd <= args.p_at).mean())

    # Success curve
    iou_thrs = np.arange(0.0, 1.0 + 1e-9, args.iou_step, dtype=float)
    success = np.array([(iou >= t).mean() for t in iou_thrs], dtype=float)

    trap = np.trapz if hasattr(np, "trapz") else np.trapezoid
    auc = float(trap(success, iou_thrs) / (iou_thrs[-1] - iou_thrs[0]))

    # Style
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "lines.linewidth": 3.0,
        "figure.dpi": 120,
        "savefig.dpi": 250,
    })

    # Precision plot
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(dist_thrs, precision, label=f"Matched-only P@{args.p_at:g}px = {p_at_val:.3f}")
    plt.xlim(0, args.max_dist)
    plt.ylim(0, 1.0)
    plt.xlabel("Location error threshold (pixels)")
    plt.ylabel("Precision rate")
    plt.title("Precision plots (matched-only)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    prec_png = f"{args.out_prefix}_precision.png"
    plt.savefig(prec_png)
    plt.close()

    # Success plot
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(iou_thrs, success, label=f"Matched-only AUC = {auc:.3f}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel("Overlap threshold (IoU)")
    plt.ylabel("Success rate")
    plt.title("Success plots (matched-only)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    succ_png = f"{args.out_prefix}_success.png"
    plt.savefig(succ_png)
    plt.close()

    print("Matched-only evaluation:")
    print(f"  matched frames used: {n2} (out of {n})")
    print(f"  P@{args.p_at:g}px = {p_at_val:.3f}")
    print(f"  AUC = {auc:.3f}")
    print("Saved:")
    print(" ", prec_png)
    print(" ", succ_png)


if __name__ == "__main__":
    main()
