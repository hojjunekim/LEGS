"""Refit the iPhone -> RealSense color correction matrix from the chart photos.

Inputs are one photo of a 24-patch ColorChecker Classic taken with the iPhone
(source) and one 640x480 color frame of the same chart from the RealSense D435
(target), both under the same lighting:

    data/calibration/color_chart/iphone.jpg
    data/calibration/color_chart/realsense.jpg

Pipeline:
  1. Detect the chart in each image with cv2.mcc (opencv-contrib, MCC24).
  2. Warp the patch grid to a 600x400 rectangle and take the per-channel
     median of the central 30% of each patch, in standard reading order
     (row-major, dark skin first).
  3. Undo the sRGB gamma on both sets of medians.
  4. Least squares for M (3x3) in  rgb_lin_realsense ~= rgb_lin_iphone @ M.

The refit is compared against M_IPHONE_TO_REALSENSE in
scripts/main/renderer/iphone_to_realsense.py. That matrix is stored rounded
to 8 decimals, so a faithful refit differs by at most 5e-9 per element.
Per-patch error is CIE76 Delta E in Lab (sRGB, D65), predicted vs RealSense.

Usage:
    python scripts/calibration/fit_color_matrix.py
    python scripts/calibration/fit_color_matrix.py \
        --iphone my_chart_iphone.jpg --realsense my_chart_d435.jpg --out ccm.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_CHART_DIR = _REPO / "data" / "calibration" / "color_chart"
sys.path.insert(0, str(_REPO / "scripts" / "main"))

from renderer.iphone_to_realsense import M_IPHONE_TO_REALSENSE  # noqa: E402

ROWS, COLS = 4, 6
PATCH_NAMES = [
    "dark skin", "light skin", "blue sky", "foliage", "blue flower", "bluish green",
    "orange", "purplish blue", "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
    "white 9.5", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5", "black 2",
]


# ----------------------------- sRGB <-> linear -----------------------------

def srgb_to_linear(srgb_01):
    s = np.asarray(srgb_01, dtype=np.float64)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin_01):
    l = np.clip(np.asarray(lin_01, dtype=np.float64), 0, None)
    out = np.where(l <= 0.0031308, l * 12.92, 1.055 * (l ** (1 / 2.4)) - 0.055)
    return np.clip(out, 0, 1)


# ----------------------------- patch extraction -----------------------------

def detect_chart_box(img_bgr):
    """Return the 4 corners (TL, TR, BR, BL) of the patch-grid bounding box."""
    if not hasattr(cv2, "mcc"):
        raise RuntimeError("cv2.mcc not found: install opencv with the contrib modules")
    det = cv2.mcc.CCheckerDetector.create()
    if not det.process(img_bgr, cv2.mcc.MCC24, 1):
        raise RuntimeError("cv2.mcc failed to detect the ColorChecker")
    return np.asarray(det.getBestColorChecker().getBox(), dtype=np.float32)


def warp_chart(img_bgr, corners, w=600, h=400):
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img_bgr, H, (w, h))


def sample_patch_medians(warped_bgr, sample_frac=0.3):
    """(24, 3) per-patch RGB medians (0-255) in reading order."""
    h, w = warped_bgr.shape[:2]
    xs = np.linspace(0, 1, 2 * COLS + 1)[1::2] * w
    ys = np.linspace(0, 1, 2 * ROWS + 1)[1::2] * h
    sx = max(2, int(w / COLS * sample_frac * 0.5))
    sy = max(2, int(h / ROWS * sample_frac * 0.5))
    medians = []
    for cy in ys:
        for cx in xs:
            cx_i, cy_i = int(round(cx)), int(round(cy))
            patch = warped_bgr[cy_i - sy:cy_i + sy + 1, cx_i - sx:cx_i + sx + 1]
            medians.append(np.median(patch.reshape(-1, 3), 0)[::-1])  # BGR -> RGB
    return np.array(medians)


def extract_patch_rgb(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {img_path}")
    return sample_patch_medians(warp_chart(img, detect_chart_box(img)))


# ----------------------------- fitting / error -----------------------------

def fit_linear_3x3(X_lin, Y_lin):
    """M (3x3) minimizing ||X @ M - Y||_F; apply as Y_pred = X @ M."""
    M, *_ = np.linalg.lstsq(X_lin, Y_lin, rcond=None)
    return M


def fit_affine_3x4(X_lin, Y_lin):
    A = np.hstack([X_lin, np.ones((len(X_lin), 1))])
    M, *_ = np.linalg.lstsq(A, Y_lin, rcond=None)
    return lambda X: np.hstack([X, np.ones((len(X), 1))]) @ M


def fit_root_poly_13(X_lin, Y_lin):
    """Finlayson et al. (2015) root-polynomial expansion, 13 terms."""
    def feats(X):
        r, g, b = X[:, 0], X[:, 1], X[:, 2]
        return np.stack([
            r, g, b, np.sqrt(r * g), np.sqrt(g * b), np.sqrt(r * b), np.cbrt(r * g * b),
            np.cbrt(r * r * g), np.cbrt(r * r * b), np.cbrt(g * g * r),
            np.cbrt(g * g * b), np.cbrt(b * b * r), np.cbrt(b * b * g),
        ], axis=1)
    M, *_ = np.linalg.lstsq(feats(X_lin), Y_lin, rcond=None)
    return lambda X: feats(X) @ M


def lab_from_srgb01(srgb_01):
    arr = np.clip(srgb_01.reshape(-1, 1, 3).astype(np.float32), 0, 1)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def delta_e(pred_lin, target_lin):
    """Per-patch CIE76 Delta E; predictions are clipped to [0, 1] first."""
    pred = linear_to_srgb(np.clip(pred_lin, 0, 1))
    target = linear_to_srgb(target_lin)
    return np.linalg.norm(lab_from_srgb01(pred) - lab_from_srgb01(target), axis=1)


def _fmt(M):
    return "\n".join("    [" + ", ".join(f"{v: .8f}" for v in row) + "]" for row in M)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iphone", default=str(_CHART_DIR / "iphone.jpg"),
                    help="chart photo from the iPhone (source)")
    ap.add_argument("--realsense", default=str(_CHART_DIR / "realsense.jpg"),
                    help="chart frame from the RealSense color stream (target)")
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="max abs element difference allowed vs the shipped matrix")
    ap.add_argument("--compare-models", action="store_true",
                    help="also report affine 3x4 and root-polynomial fits (not used)")
    ap.add_argument("--out", default=None, help="optional JSON with the refit and patch data")
    args = ap.parse_args()

    ip_med = extract_patch_rgb(args.iphone)
    rs_med = extract_patch_rgb(args.realsense)
    ip_lin = srgb_to_linear(ip_med / 255.0)
    rs_lin = srgb_to_linear(rs_med / 255.0)

    M = fit_linear_3x3(ip_lin, rs_lin)
    M_ship = np.asarray(M_IPHONE_TO_REALSENSE, dtype=np.float64)
    diff = float(np.abs(M - M_ship).max())

    dE_raw = delta_e(ip_lin, rs_lin)
    dE_fit = delta_e(ip_lin @ M, rs_lin)
    dE_ship = delta_e(ip_lin @ M_ship, rs_lin)

    print("refit M (rgb_lin_realsense = rgb_lin_iphone @ M):")
    print(_fmt(M))
    print("shipped M_IPHONE_TO_REALSENSE:")
    print(_fmt(M_ship))
    print(f"max |refit - shipped| = {diff:.3e}  (tol {args.tol:.0e})")

    print(f"\n{'#':>2}  {'patch':<14} {'iPhone RGB':>15} {'RealSense RGB':>15} "
          f"{'dE raw':>7} {'dE refit':>8} {'dE shipped':>10}")
    for i, name in enumerate(PATCH_NAMES):
        ip = " ".join(f"{v:4.0f}" for v in ip_med[i])
        rs = " ".join(f"{v:4.0f}" for v in rs_med[i])
        print(f"{i + 1:2d}  {name:<14} {ip:>15} {rs:>15} "
              f"{dE_raw[i]:7.2f} {dE_fit[i]:8.2f} {dE_ship[i]:10.2f}")
    print(f"    {'mean':<14} {'':>15} {'':>15} {dE_raw.mean():7.2f} {dE_fit.mean():8.2f} {dE_ship.mean():10.2f}")
    print(f"    {'max':<14} {'':>15} {'':>15} {dE_raw.max():7.2f} {dE_fit.max():8.2f} {dE_ship.max():10.2f}")

    models = {}
    if args.compare_models:
        models = {
            "linear_3x3": dE_fit,
            "affine_3x4": delta_e(fit_affine_3x4(ip_lin, rs_lin)(ip_lin), rs_lin),
            "root_poly_13": delta_e(fit_root_poly_13(ip_lin, rs_lin)(ip_lin), rs_lin),
        }
        print("\nmodel          dE mean  dE max   (fit and scored on the same 24 patches)")
        for k, v in models.items():
            print(f"{k:<14} {v.mean():7.2f} {v.max():7.2f}")

    if args.out:
        out = {
            "M": M.tolist(),
            "max_abs_diff_vs_shipped": diff,
            "iphone_patch_median_rgb": ip_med.tolist(),
            "realsense_patch_median_rgb": rs_med.tolist(),
            "deltaE_raw": dE_raw.tolist(),
            "deltaE_refit": dE_fit.tolist(),
            "deltaE_shipped": dE_ship.tolist(),
        }
        out.update({f"deltaE_{k}": v.tolist() for k, v in models.items()})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")

    defaults = (Path(args.iphone).resolve() == (_CHART_DIR / "iphone.jpg").resolve()
                and Path(args.realsense).resolve() == (_CHART_DIR / "realsense.jpg").resolve())
    if not defaults:
        return  # a new chart pair: the refit is the result, not a check
    if diff > args.tol:
        print(f"\nFAIL: refit differs from the shipped matrix by {diff:.3e} > {args.tol:.0e}. "
              "A cv2.mcc build that places the chart box differently changes the "
              "patch medians and therefore M.")
        sys.exit(1)
    print("\nOK: refit reproduces the shipped matrix.")


if __name__ == "__main__":
    main()
