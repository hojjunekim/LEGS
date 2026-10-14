"""Interactive click picker for SAM v2 prompts.

Opens an image in a matplotlib window. Left-click adds a positive point
(green), right-click adds a negative point (red), middle-click removes the
last point. Close the window when done.

If `--out` is given, runs sam_mask immediately on the picked points and
writes the usual artifacts (mask.png, mask_overlay.png, click.json).

Usage:
    # just print the points to paste into another command
    python pick_clicks.py --image foo.jpg

    # pick + run sam_mask in one shot
    python pick_clicks.py --image foo.jpg --out data/objects/sam3d_runs/plate_iphone
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def pick(image_path: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    import matplotlib.pyplot as plt

    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(img)
    ax.set_title(
        "left-click = positive (green) · right-click = negative (red) · "
        "middle-click = undo · close window when done"
    )
    ax.set_xlabel(f"image {W} x {H}")

    positive: list[tuple[int, int]] = []
    negative: list[tuple[int, int]] = []
    artists: list = []  # parallel to positive+negative for undo

    def redraw_status():
        fig.suptitle(f"{len(positive)} positive · {len(negative)} negative")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if event.button == 1:  # left
            positive.append((x, y))
            (h,) = ax.plot(x, y, "o", mfc="lime", mec="black", ms=12, mew=1.5)
            artists.append(("+", h))
            print(f"  + ({x}, {y})")
        elif event.button == 3:  # right
            negative.append((x, y))
            (h,) = ax.plot(x, y, "o", mfc="red", mec="black", ms=12, mew=1.5)
            artists.append(("-", h))
            print(f"  - ({x}, {y})")
        elif event.button == 2:  # middle = undo
            if not artists:
                return
            kind, h = artists.pop()
            h.remove()
            (positive if kind == "+" else negative).pop()
            print("  undo")
        redraw_status()

    fig.canvas.mpl_connect("button_press_event", on_click)
    redraw_status()
    plt.tight_layout()
    plt.show()
    return positive, negative


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None,
                    help="if set, run sam_mask on the picked points and write artifacts here")
    args = ap.parse_args()

    positive, negative = pick(args.image)
    if not positive:
        sys.exit("no positive points picked — aborting")

    pos_args = " ".join(f"--click {x},{y}" for x, y in positive)
    neg_args = " ".join(f"--neg-click {x},{y}" for x, y in negative)
    print("\npoints:")
    print(f"  positive = {positive}")
    print(f"  negative = {negative}")
    print(f"\nequivalent CLI: {pos_args} {neg_args}".rstrip())

    if args.out:
        _HERE = Path(__file__).resolve().parent
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        import sam_mask  # noqa: E402
        print(f"\nrunning sam_mask -> {args.out}")
        sam_mask.run(args.image, positive, args.out, negative=negative)


if __name__ == "__main__":
    main()
