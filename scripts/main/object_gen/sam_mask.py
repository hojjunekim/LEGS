"""Step 1: iPhone JPG + click point(s) -> binary 2D mask via Meta SAM v2.

Outputs (under <out_dir>):
    iphone.jpg          # input copy (re-encoded as JPEG)
    click.json          # {"positive": [[x,y], ...], "negative": [[x,y], ...], "image_size": [W,H]}
    mask.png            # uint8 0/255, same HxW as input
    mask_overlay.png    # mask blended onto iphone.jpg for sanity check

Usage:
    # single positive click (legacy)
    python sam_mask.py --image foo.jpg --click 612,840 --out runs/mug

    # multi-point: center + rim (e.g. plate with patterned rim that SAM splits off)
    python sam_mask.py --image plate.jpg --click 2050,3550 --click 1300,3550 --out runs/plate

    # mix positive + negative (e.g. exclude a glare highlight)
    python sam_mask.py --image foo.jpg --click 612,840 --neg-click 700,820 --out runs/mug
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_click(s: str) -> tuple[int, int]:
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expects 'x,y', got {s!r}")
    return int(parts[0]), int(parts[1])


def predict_mask(
    image_rgb: np.ndarray,
    positive: list[tuple[int, int]],
    negative: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Run SAM v2 with N positive + M negative point prompts; return uint8 0/255 mask."""
    # Imported inside so that --help works without the heavy dep installed.
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    negative = negative or []
    if not positive:
        raise ValueError("need at least one positive click")

    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
    predictor.set_image(image_rgb)
    pts = np.asarray(list(positive) + list(negative), dtype=np.float32)
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32)
    # multimask_output is only meaningful for single-point prompts. With >=2
    # points SAM returns one mask and ranks/refines from the prompt itself.
    multimask = (len(positive) + len(negative)) == 1
    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=labels,
        multimask_output=multimask,
    )
    best = int(np.argmax(scores))
    mask_bool = masks[best].astype(bool)
    print(f"  SAM2 picked mask {best}/{len(scores)} (score={scores[best]:.3f}) "
          f"from {len(positive)} pos + {len(negative)} neg points")
    return (mask_bool.astype(np.uint8) * 255)


def write_overlay(
    out_path: Path,
    image_rgb: np.ndarray,
    mask_u8: np.ndarray,
    positive: list[tuple[int, int]],
    negative: list[tuple[int, int]] | None = None,
) -> None:
    """Red-tint mask over image + green crosshairs at positive clicks, blue at negative."""
    overlay = image_rgb.astype(np.float32).copy()
    sel = mask_u8 > 0
    overlay[sel] = overlay[sel] * 0.55 + np.array([255, 50, 50], dtype=np.float32) * 0.45
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    H, W = overlay.shape[:2]

    def _draw_cross(cx, cy, color):
        if not (0 <= cy < H and 0 <= cx < W):
            return
        r = 6
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        overlay[y0:y1, max(0, cx - 1):min(W, cx + 2)] = color
        overlay[max(0, cy - 1):min(H, cy + 2), x0:x1] = color

    for cx, cy in positive:
        _draw_cross(cx, cy, [0, 255, 0])
    for cx, cy in negative or []:
        _draw_cross(cx, cy, [0, 100, 255])
    Image.fromarray(overlay).save(out_path)


def run(
    image_path: str,
    click_xy,
    out_dir: str,
    negative: list[tuple[int, int]] | None = None,
) -> dict:
    """`click_xy` is either a single (x,y) tuple or a list of (x,y) tuples."""
    if click_xy and isinstance(click_xy[0], (int, float)):
        positive = [tuple(click_xy)]  # legacy single-point call
    else:
        positive = [tuple(p) for p in click_xy]
    negative = [tuple(p) for p in (negative or [])]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    image_rgb = np.array(img)
    H, W = image_rgb.shape[:2]
    for x, y in positive + negative:
        if not (0 <= x < W and 0 <= y < H):
            raise ValueError(f"click ({x},{y}) out of bounds for image {W}x{H}")

    iphone_out = out / "iphone.jpg"
    img.save(iphone_out, quality=95)

    mask_u8 = predict_mask(image_rgb, positive, negative)
    Image.fromarray(mask_u8).save(out / "mask.png")
    write_overlay(out / "mask_overlay.png", image_rgb, mask_u8, positive, negative)

    meta = {
        "positive": [[int(x), int(y)] for x, y in positive],
        "negative": [[int(x), int(y)] for x, y in negative],
        "image_size": [W, H],
    }
    with open(out / "click.json", "w") as f:
        json.dump(meta, f, indent=2)

    coverage = float(mask_u8.mean()) / 255.0
    print(f"  mask coverage = {coverage * 100:.2f}% of frame")
    return {"mask": str(out / "mask.png"), "overlay": str(out / "mask_overlay.png")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="path to iPhone JPG/PNG")
    p.add_argument("--click", required=True, type=_parse_click, action="append",
                   help="positive point 'x,y' (repeatable for multi-point prompts)")
    p.add_argument("--neg-click", type=_parse_click, action="append", default=[],
                   help="negative point 'x,y' (repeatable; SAM excludes this region)")
    p.add_argument("--out", required=True, help="output dir")
    args = p.parse_args()
    run(args.image, args.click, args.out, negative=args.neg_click)


if __name__ == "__main__":
    main()
