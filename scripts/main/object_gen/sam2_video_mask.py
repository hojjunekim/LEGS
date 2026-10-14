"""SAM2 per-object video mask: mp4 + click(s) on frame 0 -> one mp4 per object.

For each `--object NAME`, opens a click GUI on frame 0 (left-click = foreground,
right-click = background). After clicks, a preview popup shows the frame-0
green-on-black silhouette for that object; close to accept, click Reject to
re-click. Once all objects are accepted, SAM2VideoPredictor propagates the full
video in a single pass (all objects share one image-feature pass), and each
object is written to `<out-dir>/<prefix>_<NAME>.mp4` (or `<NAME>.mp4` if prefix
is empty). Pixels outside the per-object mask are black; mask pixels are pure
green.

Run inside the sam3d-objects conda env (has sam2 + matplotlib + opencv).

Usage:
    python sam2_video_mask.py \\
        --video clip.mp4 \\
        --out-dir masked \\
        --prefix clip \\
        --object robot --object orange
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read first frame of {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def collect_clicks_gui(
    frame_rgb: np.ndarray, label: str = "object"
) -> tuple[list, list]:
    import matplotlib.pyplot as plt

    pos, neg = [], []
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(frame_rgb)
    ax.set_title(
        f"clicks for '{label}': left=foreground, right=background "
        "(close window when done)"
    )

    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:
            pos.append((x, y))
            ax.plot(x, y, "o", color="lime", markersize=10, markeredgecolor="black")
        elif event.button == 3:
            neg.append((x, y))
            ax.plot(x, y, "x", color="red", markersize=12, markeredgewidth=3)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    return pos, neg


def show_preview_window(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    positive: list[tuple[int, int]],
    negative: list[tuple[int, int]],
    label: str = "object",
) -> bool:
    """Frame-0 mask preview. Close window = accept. 'Reject' button = redo."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    green = np.zeros_like(frame_rgb)
    green[mask > 0] = (0, 255, 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(frame_rgb)
    axes[0].set_title(f"'{label}' clicks")
    for x, y in positive:
        axes[0].plot(x, y, "o", color="lime", markersize=10, markeredgecolor="black")
    for x, y in negative:
        axes[0].plot(x, y, "x", color="red", markersize=12, markeredgewidth=3)
    axes[1].imshow(green)
    axes[1].set_title(f"'{label}' frame-0 green silhouette")
    fig.suptitle(
        "close window to ACCEPT and continue, or click Reject to re-click this object"
    )

    state = {"accept": True}

    def on_reject(_event):
        state["accept"] = False
        plt.close(fig)

    bax = fig.add_axes([0.45, 0.02, 0.12, 0.05])
    btn = Button(bax, "Reject & Re-click")
    btn.on_clicked(on_reject)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()
    return state["accept"]


def predict_masks_multi(
    video_path: Path,
    objects: list[dict],
    frame0_rgb: np.ndarray,
    skip_preview: bool = False,
) -> dict[int, dict[int, np.ndarray]]:
    """Returns {obj_id: {frame_idx: mask}}. Mutates `objects` (pos/neg/accepted)."""
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained(
        "facebook/sam2.1-hiera-large", device=device
    )

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(video_path))

        i = 0
        while i < len(objects):
            obj = objects[i]
            obj_id = i + 1

            if obj["pos"] is None:
                print(f"  [obj {obj_id}: '{obj['name']}'] click GUI")
                pos, neg = collect_clicks_gui(frame0_rgb, label=obj["name"])
                if not pos:
                    raise SystemExit(
                        f"need at least one positive click for '{obj['name']}'"
                    )
                obj["pos"] = pos
                obj["neg"] = neg

            points = np.asarray(
                list(obj["pos"]) + list(obj["neg"]), dtype=np.float32
            )
            labels = np.asarray(
                [1] * len(obj["pos"]) + [0] * len(obj["neg"]), dtype=np.int32
            )
            _, out_ids, mlogits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
            ch = list(out_ids).index(obj_id)
            m0 = (mlogits[ch] > 0).squeeze(0).to(torch.uint8).cpu().numpy() * 255

            if skip_preview or obj["accepted"]:
                obj["accepted"] = True
                i += 1
                continue

            print(
                f"  [{obj['name']}] preview: "
                f"{len(obj['pos'])} fg + {len(obj['neg'])} bg click(s)"
            )
            accepted = show_preview_window(
                frame0_rgb, m0, obj["pos"], obj["neg"], label=obj["name"]
            )
            if accepted:
                obj["accepted"] = True
                i += 1
                continue

            print(f"  [{obj['name']}] rejected, re-clicking...")
            obj["pos"] = None
            obj["neg"] = None
            obj["accepted"] = False
            predictor.reset_state(state)
            for prev in objects[:i]:
                prev["accepted"] = True
            i = 0

        print(f"  propagating {len(objects)} object(s) across full video...")
        all_masks: dict[int, dict[int, np.ndarray]] = {
            i + 1: {} for i in range(len(objects))
        }
        for fidx, out_ids, mlogits in predictor.propagate_in_video(state):
            for ch, oid in enumerate(out_ids):
                m = (mlogits[ch] > 0).squeeze(0).to(torch.uint8).cpu().numpy() * 255
                all_masks[int(oid)][fidx] = m
        return all_masks


def composite_video(
    video_path: Path,
    masks: dict[int, np.ndarray],
    out_path: Path,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )
    GREEN_BGR = (0, 255, 0)
    i = 0
    while True:
        ok, _frame_bgr = cap.read()
        if not ok:
            break
        out_bgr = np.zeros((H, W, 3), dtype=np.uint8)
        m = masks.get(i)
        if m is not None:
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            out_bgr[m > 0] = GREEN_BGR
        writer.write(out_bgr)
        i += 1
    cap.release()
    writer.release()
    print(f"  wrote {out_path}  ({i} frames @ {fps:.2f} fps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--prefix",
        type=str,
        default="",
        help="output filename prefix; output is <out-dir>/<prefix>_<object>.mp4",
    )
    ap.add_argument(
        "--object",
        action="append",
        required=True,
        help="object label; opens a click GUI per --object. repeatable.",
    )
    ap.add_argument(
        "--skip-preview",
        action="store_true",
        help="propagate immediately; do not show frame-0 confirmation popups",
    )
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    frame0 = read_first_frame(args.video)
    objects = [
        {"name": label, "pos": None, "neg": None, "accepted": False}
        for label in args.object
    ]

    print(
        f"[{args.video.name}] {len(objects)} object(s): "
        f"{[o['name'] for o in objects]}"
    )

    all_masks = predict_masks_multi(
        args.video, objects, frame0, skip_preview=args.skip_preview
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, obj in enumerate(objects):
        obj_id = i + 1
        sep = "_" if args.prefix else ""
        out_name = f"{args.prefix}{sep}{obj['name']}.mp4"
        out_path = args.out_dir / out_name
        composite_video(args.video, all_masks[obj_id], out_path)


if __name__ == "__main__":
    main()
