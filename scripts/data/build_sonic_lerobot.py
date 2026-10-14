"""
Build a universal LeRobot v2.1 dataset (SONIC + Psi0 columns) from sim
collection.

Inputs (produced upstream):
    1. ``<data_dir>/episode_XXXX.json`` and ``episode_XXXX_scenario.json``
       — written by ``DataRecorder`` during ``simulate_object.py`` runs.
    2. ``<data_dir>/final_renders/episode_XXXX/<aug_idx:02d>/<frame:05d>_d435.jpg``
       (or ``.png``) — written by ``offline_renderer.py``.
    3. ``<token_csv_dir>/.../token_state.csv``
       — written by ``deploy.sh sim ... --enable-csv-logs --logs-dir
       <token_csv_dir>``. Optional. If absent, ``action.motion_token`` is
       zero-filled (fine for Psi0 fine-tune; required for SONIC encoder
       fine-tune).

Output:
    A LeRobot v2.1 dataset at ``<save_root>`` containing:
      data/chunk-XXX/episode_XXXXXX.parquet — full SONIC schema + Psi0 cols
      videos/chunk-XXX/observation.images.ego_view/episode_XXXXXX.mp4
      meta/info.json, meta/episodes.jsonl, meta/episodes_stats.jsonl
      meta/tasks.jsonl
      meta/modality.json              — default = SONIC
      meta/modality_gr00t_sonic.json  — explicit alias of meta/modality.json
      meta/modality_psi0.json         — Psi0 18-dim ROS2 head

Usage (run with ``submodules/Psi0/.venv-openpi`` (lerobot 0.1.0) after
installing ``gear_sonic`` into that env; see SETUP.md):
    PSI_VENV=submodules/Psi0/.venv-openpi
    VIRTUAL_ENV=$PSI_VENV uv pip install -e submodules/GR00T-WholeBodyControl/gear_sonic --no-deps
    $PSI_VENV/bin/python scripts/data/build_sonic_lerobot.py \
        --data-dir data/collected/<run> \
        --save-root data/lerobot/wigs/<name>_uni \
        --token-csv-dir data/collected/<run>/csv_snapshot

Every episode is checked before anything is written: a missing scenario
JSON, render dir or rendered frame stops the build unless ``--allow-skip``
(leave such episodes out) or ``--allow-missing-frames`` (fill black) is set.

The script joins the deploy CSV by the host ``CLOCK_REALTIME`` timestamp
recorded in each frame (see ``DataRecorder`` ``time_realtime_ms``).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# utils.sonic_frame_builder lives in scripts/main/utils (kept there to avoid
# splitting the shared utils package). Add scripts/main to sys.path so the
# import resolves whether this script is launched from scripts/data or elsewhere.
MAIN_DIR = Path(__file__).resolve().parent.parent / "main"
sys.path.insert(0, str(MAIN_DIR))
from utils.sonic_frame_builder import (  # noqa: E402
    EGO_VIEW_HEIGHT,
    EGO_VIEW_WIDTH,
    build_sonic_frame,
    get_modality_config_psi0,
    get_modality_config_sonic,
    get_universal_features,
)

# gear_sonic must be importable: ``uv pip install -e <gear_sonic_path> --no-deps``
from gear_sonic.data.exporter import Gr00tDataExporter  # noqa: E402


FPS = 30  # matches simulate_object.py loop rate
DEFAULT_AUG_SUBDIR = "00"


# ---------------------------------------------------------------------------
# Token CSV ingestion
# ---------------------------------------------------------------------------


def discover_token_csvs(token_csv_dir: str | None) -> list[Path]:
    """Find every ``token_state.csv`` under ``token_csv_dir`` (recursively)."""
    if not token_csv_dir:
        return []
    root = Path(token_csv_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("token_state.csv"))


def load_concat_tokens(csv_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every ``token_state.csv`` into ``(timestamps_ms, tokens)``
    sorted by timestamp. Empty files / parse errors are skipped with a warning.
    """
    if not csv_paths:
        return np.array([], dtype=np.float64), np.zeros((0, 64), dtype=np.float64)

    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  [tokens] skip {p}: {e}")
            continue
        token_cols = [c for c in df.columns if c.startswith("token_")]
        if "time_realtime_ms" not in df.columns or not token_cols:
            print(f"  [tokens] skip {p}: missing time_realtime_ms or token_* columns")
            continue
        token_cols = sorted(token_cols, key=lambda c: int(c.split("_")[1]))
        frames.append(df[["time_realtime_ms"] + token_cols].values)

    if not frames:
        return np.array([], dtype=np.float64), np.zeros((0, 64), dtype=np.float64)

    arr = np.concatenate(frames, axis=0)
    arr = arr[np.argsort(arr[:, 0])]
    timestamps = arr[:, 0].astype(np.float64)
    tokens = arr[:, 1:].astype(np.float64)
    if tokens.shape[1] != 64:
        print(f"  [tokens] WARNING: expected 64-d motion_token, got {tokens.shape[1]}")
    return timestamps, tokens


def lookup_token(
    timestamps: np.ndarray,
    tokens: np.ndarray,
    target_ms: float,
    max_delta_ms: float = 50.0,
) -> np.ndarray | None:
    """Nearest-neighbor lookup. Returns None if the closest CSV row exceeds
    ``max_delta_ms`` — caller can decide to zero-fill or drop the frame."""
    if timestamps.size == 0:
        return None
    idx = int(np.argmin(np.abs(timestamps - target_ms)))
    if abs(timestamps[idx] - target_ms) > max_delta_ms:
        return None
    return tokens[idx]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def frame_path(render_dir: Path, frame_idx: int) -> Path | None:
    """Rendered d435 frame ``frame_idx`` (JPEG preferred over PNG), or None."""
    for ext in ("jpg", "png"):
        path = render_dir / f"{frame_idx:05d}_d435.{ext}"
        if path.exists():
            return path
    return None


def load_rgb_or_black(path: Path | None) -> np.ndarray:
    """Read PNG/JPEG → uint8 RGB, resized to (H=480, W=640) if needed.
    Fallback black frame (only with ``--allow-missing-frames``) keeps the
    parquet schema valid when offline_renderer skipped a frame.
    """
    if path is not None and path.exists():
        img = np.array(Image.open(path).convert("RGB"))
        if img.shape[:2] != (EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH):
            img = np.array(
                Image.fromarray(img).resize((EGO_VIEW_WIDTH, EGO_VIEW_HEIGHT))
            )
        return img.astype(np.uint8)
    return np.zeros((EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Episode iteration
# ---------------------------------------------------------------------------


def find_episodes(data_dir: Path) -> list[str]:
    """Return the four-digit episode index strings present in ``data_dir``."""
    files = sorted(data_dir.glob("episode_[0-9][0-9][0-9][0-9].json"))
    return [f.stem.replace("episode_", "") for f in files]


def add_episode(
    exporter: Gr00tDataExporter,
    records: list[dict],
    scenario: dict,
    render_dir: Path,
    timestamps: np.ndarray,
    tokens: np.ndarray,
    token_max_delta_ms: float,
    task_override: str | None,
) -> int:
    """Add every frame of one episode to the exporter and ``save_episode()``.

    Returns the number of frames written. The exporter's video_writer encodes
    mp4 inline via PyAV; ``save_episode`` flushes + closes it.
    """
    n_written = 0
    n_token_miss = 0
    task = task_override or scenario.get("language_instruction") or ""

    for frame_idx, record in enumerate(records):
        image = load_rgb_or_black(frame_path(render_dir, frame_idx))

        ts_ms = float(record.get("time_realtime_ms", 0.0))
        token = lookup_token(timestamps, tokens, ts_ms, token_max_delta_ms)
        if token is None and timestamps.size > 0:
            n_token_miss += 1

        sonic_frame = build_sonic_frame(
            record=record,
            scenario=scenario,
            image_rgb_uint8=image,
            motion_token=token,
            task=task,
        )

        exporter.add_frame(sonic_frame)
        n_written += 1

    if n_token_miss > 0:
        print(f"    [tokens] {n_token_miss}/{n_written} frames had no CSV match within tolerance — zero-filled")
    exporter.save_episode()
    return n_written


# ---------------------------------------------------------------------------
# Modality sidecars
# ---------------------------------------------------------------------------


def write_modality_sidecars(save_root: Path) -> None:
    """The exporter wrote ``meta/modality.json`` (= SONIC) at create time.
    Add the Psi0 modality + an explicit-name SONIC alias next to it."""
    meta_dir = save_root / "meta"
    psi0_path = meta_dir / "modality_psi0.json"
    sonic_path = meta_dir / "modality_gr00t_sonic.json"

    with open(psi0_path, "w") as f:
        json.dump(get_modality_config_psi0(), f, indent=4)
    with open(sonic_path, "w") as f:
        json.dump(get_modality_config_sonic(), f, indent=4)
    print(f"  Wrote {psi0_path.name} and {sonic_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Build universal SONIC + Psi0 LeRobot dataset")
    # Single-source mode (backwards compat with wigs2universal.sh / sam2universal.sh).
    parser.add_argument("--data-dir", default=None,
                        help="(single-source) Dir with episode_*.json scenario JSON and renders/")
    parser.add_argument("--task", default=None,
                        help="(single-source) Override per-episode language_instruction with a fixed string")
    # Multi-source mode (paired by index — emits >=2 task indices).
    # Each --source-* arg is repeatable; lists must have equal length so the
    # i-th element of each pairs into one (data_dir, ep_from, ep_to, task) tuple.
    # Episodes from each source are included if ep_from <= int(ep_str) <= ep_to.
    parser.add_argument("--source-data-dir", action="append", default=[],
                        help="(multi-source) Repeatable: source dir for one slice")
    parser.add_argument("--source-ep-from", action="append", default=[], type=int,
                        help="(multi-source) Repeatable: inclusive lower ep index for the paired source")
    parser.add_argument("--source-ep-to", action="append", default=[], type=int,
                        help="(multi-source) Repeatable: inclusive upper ep index for the paired source")
    parser.add_argument("--source-task", action="append", default=[],
                        help="(multi-source) Repeatable: language_instruction for the paired source")
    parser.add_argument("--save-root", required=True,
                        help="Output dataset root (LeRobot v2.1 layout)")
    parser.add_argument("--token-csv-dir", default=None,
                        help="Deploy logs dir (recursively scanned for token_state.csv). "
                             "Omit to zero-fill action.motion_token.")
    parser.add_argument("--render-subdir", default=DEFAULT_AUG_SUBDIR,
                        help=f"Per-episode render index under <renders-dir>/episode_XXXX/ "
                             f"(default '{DEFAULT_AUG_SUBDIR}', use augmentation indices for aug)")
    parser.add_argument("--renders-dir-name", default="final_renders",
                        help="Subdir under each source data-dir holding pre-rendered frames "
                             "(default 'final_renders'; use 'baseline_renders' for the "
                             "mesh-only baseline produced by offline_renderer_baseline.py).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Wipe save_root before writing (default: resume / append)")
    parser.add_argument("--allow-skip", action="store_true",
                        help="Leave out episodes with no scenario JSON or render dir "
                             "(default: stop before writing anything)")
    parser.add_argument("--allow-missing-frames", action="store_true",
                        help="Fill missing rendered frames with black "
                             "(default: stop before writing anything)")
    parser.add_argument("--token-max-delta-ms", type=float, default=50.0,
                        help="Reject CSV match if nearest-neighbor distance exceeds this (ms)")
    return parser.parse_args()


def main():
    args = parse_args()
    save_root = Path(os.path.expanduser(args.save_root)).resolve()

    # Resolve source list: multi-source if --source-data-dir was supplied,
    # otherwise wrap the single --data-dir / --task pair as a one-element list.
    if args.source_data_dir:
        n_src = len(args.source_data_dir)
        if not (n_src == len(args.source_ep_from) == len(args.source_ep_to) == len(args.source_task)):
            sys.exit("--source-data-dir / --source-ep-from / --source-ep-to / --source-task "
                     "lists must have equal length")
        sources = [
            (Path(d).resolve(), int(ef), int(et), task)
            for d, ef, et, task in zip(args.source_data_dir, args.source_ep_from,
                                       args.source_ep_to, args.source_task)
        ]
    elif args.data_dir:
        sources = [(Path(args.data_dir).resolve(), 0, 10**9, args.task or "")]
    else:
        sys.exit("Provide --data-dir (single-source) or --source-data-dir (multi-source)")

    for d, _ef, _et, _t in sources:
        if not d.is_dir():
            sys.exit(f"data-dir not found: {d}")
        if not (d / args.renders_dir_name).is_dir():
            sys.exit(f"Expected pre-rendered frames at {d / args.renders_dir_name}. "
                     f"Run scripts/data/offline_renderer.py (or _baseline.py) first.")

    # Build flat ordered list of (data_dir, ep_str, task) entries that fall
    # within the requested ep ranges. Task may be "" → fall back to scenario JSON.
    entries: list[tuple[Path, str, str]] = []
    for d, ef, et, task in sources:
        for ep_str in find_episodes(d):
            ep_int = int(ep_str)
            if ef <= ep_int <= et:
                entries.append((d, ep_str, task))

    if not entries:
        sys.exit("No episodes match the requested ep ranges across all sources")
    if args.token_csv_dir and not Path(args.token_csv_dir).is_dir():
        sys.exit(f"--token-csv-dir {args.token_csv_dir} does not exist "
                 f"(omit it to zero-fill action.motion_token)")

    # Check every episode before writing anything, so a gap in the inputs
    # can't silently drop an episode or bake black frames into the dataset.
    skips, frame_gaps = [], []
    for data_dir, ep_str, _task in entries:
        tag = f"{data_dir.name}/ep {ep_str}"
        sce_path = data_dir / f"episode_{ep_str}_scenario.json"
        render_dir = data_dir / args.renders_dir_name / f"episode_{ep_str}" / args.render_subdir
        if not sce_path.exists():
            skips.append(f"{tag}: missing scenario JSON")
            continue
        if not render_dir.is_dir():
            skips.append(f"{tag}: missing renders at {render_dir}")
            continue
        with open(data_dir / f"episode_{ep_str}.json", "r") as f:
            n_records = len(json.load(f))
        n_missing = sum(frame_path(render_dir, i) is None for i in range(n_records))
        if n_missing:
            frame_gaps.append(f"{tag}: {n_missing}/{n_records} rendered frames missing in {render_dir}")
    for msg in skips + frame_gaps:
        print(f"  {msg}")
    if skips and not args.allow_skip:
        sys.exit(f"{len(skips)} episode(s) can't be built (listed above). Fix them, "
                 f"or pass --allow-skip to leave them out.")
    if frame_gaps and not args.allow_missing_frames:
        sys.exit(f"{len(frame_gaps)} episode(s) have missing rendered frames (listed above). "
                 f"Re-render them, or pass --allow-missing-frames to fill black.")

    if args.overwrite and save_root.exists():
        shutil.rmtree(save_root)
        print(f"Removed existing dataset at {save_root}")

    token_csvs = discover_token_csvs(args.token_csv_dir)
    if token_csvs:
        print(f"Loading {len(token_csvs)} token_state.csv file(s) from {args.token_csv_dir}")
    else:
        if args.token_csv_dir:
            print(f"  [tokens] no token_state.csv files under {args.token_csv_dir} — motion_token will be zero")
        else:
            print("  [tokens] --token-csv-dir not provided — motion_token will be zero (Psi0 fine-tune only)")
    timestamps, tokens = load_concat_tokens(token_csvs)
    if timestamps.size:
        print(f"  Loaded {timestamps.size} token rows spanning "
              f"{(timestamps[-1] - timestamps[0]) / 1000.0:.1f}s")

    # Default task = first entry's source task, falling back to its scenario JSON.
    first_dd, first_ep, first_task = entries[0]
    if not first_task:
        with open(first_dd / f"episode_{first_ep}_scenario.json", "r") as f:
            first_task = json.load(f).get("language_instruction") or "task"
    default_task = first_task

    print(f"Creating dataset at {save_root} with {len(sources)} source(s), "
          f"{len(entries)} total episode(s)")
    for d, ef, et, task in sources:
        n_eps = sum(1 for dd, ee, _ in entries if dd == d and ef <= int(ee) <= et)
        print(f"  source: {d} eps [{ef}..{et}] task={task!r} -> {n_eps} ep(s)")

    exporter = Gr00tDataExporter.create(
        save_root=str(save_root),
        fps=FPS,
        features=get_universal_features(),
        modality_config=get_modality_config_sonic(),  # default modality.json
        task=default_task,
        script_config={
            "source": "LEGS/scripts/data/build_sonic_lerobot.py",
            "sources": [
                {"data_dir": str(d), "ep_from": ef, "ep_to": et, "task": task}
                for d, ef, et, task in sources
            ],
            "token_csv_dir": args.token_csv_dir or "",
            "render_subdir": args.render_subdir,
            "renders_dir_name": args.renders_dir_name,
            # Only present when --allow-skip / --allow-missing-frames let a
            # gap through, so a built dataset says what it left out.
            **({"skipped_raw_episodes": skips} if skips else {}),
            **({"black_filled_frames": frame_gaps} if frame_gaps else {}),
        },
        overwrite_existing=False,  # we already handled --overwrite above
    )

    total_frames = 0
    n_written = 0
    for data_dir, ep_str, task in entries:
        render_root = data_dir / args.renders_dir_name
        rec_path = data_dir / f"episode_{ep_str}.json"
        sce_path = data_dir / f"episode_{ep_str}_scenario.json"
        render_dir = render_root / f"episode_{ep_str}" / args.render_subdir
        if not sce_path.exists():
            print(f"  [{data_dir.name}/ep {ep_str}] skip — missing scenario JSON")
            continue
        if not render_dir.is_dir():
            print(f"  [{data_dir.name}/ep {ep_str}] skip — missing renders at {render_dir}")
            continue
        with open(rec_path, "r") as f:
            records = json.load(f)
        with open(sce_path, "r") as f:
            scenario = json.load(f)

        print(f"  [{data_dir.name}/ep {ep_str}] {len(records)} frames -> ", end="", flush=True)
        n = add_episode(
            exporter=exporter,
            records=records,
            scenario=scenario,
            render_dir=render_dir,
            timestamps=timestamps,
            tokens=tokens,
            token_max_delta_ms=args.token_max_delta_ms,
            task_override=(task or None),
        )
        total_frames += n
        n_written += 1
        print(f"saved {n}")

    write_modality_sidecars(save_root)
    print(f"\nDone. {n_written} episodes, {total_frames} total frames at {save_root}")


if __name__ == "__main__":
    main()
