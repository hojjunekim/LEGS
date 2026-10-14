"""Multi-process wrapper around offline_renderer_baseline.

Splits the episode list across N worker processes, each of which runs the same
mesh-only pipeline as `offline_renderer_baseline.main` on its slice. Workers
spawn fresh ("spawn" start method), so each gets its own EGL context and its
own pyrender state — no sharing between processes.

Episodes are independent (separate output subdirs), so the only contention is
the GPU. The baseline pipeline has no gsplat, so unlike offline_renderer_mp.py
there's no JIT-load lock to serialize on.

Usage:
    python offline_renderer_baseline_mp.py --data-dir <dir> --workers 8
"""

import argparse
import multiprocessing as mp
import os
import time

# Run before pyrender imports below so children inherit the right backend.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYRENDER_OFFSCREEN", "1")


def _worker(worker_idx, robot_files, data_dir, scene_override,
            out_dir, cameras, video, fps):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYRENDER_OFFSCREEN", "1")
    import offline_renderer_baseline

    tag = f"W{worker_idx}"
    print(f"[{tag}] {len(robot_files)} episodes: "
          f"{[os.path.basename(p) for p in robot_files]}", flush=True)

    offline_renderer_baseline.render_episode_files(
        robot_files=robot_files,
        data_dir=data_dir,
        scene_override=scene_override,
        out_dir=out_dir,
        cameras=cameras,
        video=video,
        fps=fps,
        worker_tag=tag,
    )


def _split_round_robin(items, n):
    chunks = [[] for _ in range(n)]
    for i, it in enumerate(items):
        chunks[i % n].append(it)
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Episode dir (reads episode_*.json, writes baseline_renders/).")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel worker processes (default: 2).")
    parser.add_argument("--episode-from", type=int, default=None,
                        help="Skip episodes below this index.")
    parser.add_argument("--mid360", action="store_true",
                        help="Also render mid360 camera (default: d435 only).")
    parser.add_argument("--scene", type=str, default=None,
                        help="Render with this scene instead of the recorded one "
                             "(e.g. an appearance variant from scene_config.py).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output root (default <data-dir>/baseline_renders/).")
    parser.add_argument("--video", action="store_true",
                        help="Stitch the rendered JPGs into an MP4 per episode.")
    parser.add_argument("--fps", type=int, default=20,
                        help="FPS for stitched MP4 (default 20).")
    args = parser.parse_args()

    import offline_renderer_baseline
    offline_renderer_baseline.check_scene_assets(args.scene)

    cameras = ("d435", "mid360") if args.mid360 else ("d435",)

    robot_files = offline_renderer_baseline._enumerate_episode_files(
        args.data_dir, episode=None, episode_from=args.episode_from,
    )
    if not robot_files:
        raise SystemExit(f"Error: No episode data found in {args.data_dir}")

    n_workers = max(1, min(args.workers, len(robot_files)))
    chunks = _split_round_robin(robot_files, n_workers)
    print(f"[MP] {len(robot_files)} episodes split across {n_workers} workers "
          f"(chunk sizes: {[len(c) for c in chunks]})")

    ctx = mp.get_context("spawn")
    procs = []
    t_start = time.perf_counter()
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = ctx.Process(
            target=_worker,
            args=(i, chunk, args.data_dir, args.scene,
                  args.out_dir, cameras, args.video, args.fps),
            name=f"baseline-render-worker-{i}",
        )
        p.start()
        procs.append(p)

    failed = []
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((p.name, p.exitcode))

    t_elapsed = time.perf_counter() - t_start
    n_total = sum(len(c) for c in chunks)
    print(f"\n[MP] {n_workers} workers, {n_total} episodes, "
          f"{t_elapsed:.2f}s wall time "
          f"({t_elapsed / max(n_total, 1):.2f}s/episode avg).")
    if failed:
        for name, code in failed:
            print(f"  WORKER FAIL: {name} exit={code}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
