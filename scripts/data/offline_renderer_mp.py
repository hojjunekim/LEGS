"""Multi-process wrapper around offline_renderer.

Splits the episode list across N worker processes, each of which runs the same
pipeline as `offline_renderer.main` on its slice. Workers spawn fresh (the
"spawn" start method), so each gets its own EGL context, its own gsplat
VisualizerSceneManager loaded onto the GPU, and its own pyrender state — no
sharing between processes.

Episodes are independent (separate output subdirs), so the only contention is
the GPU. Use `--workers N` to dial in the right concurrency for your GPU
memory budget: each worker uses roughly ~1-3 GB on top of base CUDA overhead.

Usage:
    python offline_renderer_mp.py --data-dir <dir> --workers 3
    # appearance variant (scene_config.py); *_blue variants pick the blue splat
    python offline_renderer_mp.py --data-dir <dir> --scene apple_to_box_turn_blue --workers 3
"""

import argparse
import multiprocessing as mp
import os
import time

# Run before pyrender/torch imports below so children inherit the right backend.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYRENDER_OFFSCREEN", "1")


def _worker(worker_idx, robot_files, data_dir, ply_path,
            scene_override, cameras, enable_hsv_jitter, gsplat_lock,
            allow_uncalibrated=False):
    """Child-process entry: imports the renderer and runs its slice.

    `gsplat_lock` serializes the first gsplat CUDA-extension JIT load across
    workers. Without it, two workers race on torch's `file_baton` lockfile
    (one removes it, the other gets FileNotFoundError on its own `os.remove`).
    Holding the lock just through the lazy import of gsplat's `_backend` is
    enough — once `_C` is loaded into one process, that process is done with
    the on-disk JIT machinery and the next worker can take the lock.
    """
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["PYRENDER_OFFSCREEN"] = "1"
    import offline_renderer  # imported lazily so parent stays lightweight

    tag = f"W{worker_idx}"
    print(f"[{tag}] {len(robot_files)} episodes: "
          f"{[os.path.basename(p) for p in robot_files]}", flush=True)

    with gsplat_lock:
        print(f"[{tag}] Loading gsplat CUDA extension (serialized)...", flush=True)
        import torch  # ensure CUDA is initialised before the JIT load
        torch.cuda.init()
        from gsplat.cuda import _backend  # triggers JIT load behind the file_baton
        _ = _backend._C  # touch — confirms the extension is bound
        print(f"[{tag}] gsplat extension ready.", flush=True)

    offline_renderer.render_episode_files(
        robot_files=robot_files,
        data_dir=data_dir,
        ply_path=ply_path,
        scene_override=scene_override,
        cameras=cameras,
        enable_hsv_jitter=enable_hsv_jitter,
        worker_tag=tag,
        allow_uncalibrated=allow_uncalibrated,
    )


def _split_round_robin(items, n):
    """Round-robin shard so worker load is balanced even when episode lengths
    correlate with index. Returns n lists (may include empties if len(items)<n)."""
    chunks = [[] for _ in range(n)]
    for i, it in enumerate(items):
        chunks[i % n].append(it)
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Episode dir (reads episode_*.json, writes final_renders/).")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel worker processes (default: 2).")
    parser.add_argument("--episode-from", type=int, default=None,
                        help="Skip episodes below this index.")
    parser.add_argument("--mid360", action="store_true",
                        help="Also render mid360 camera (default: d435 only).")
    parser.add_argument("--scene", type=str, default=None,
                        help="Render with this scene instead of the recorded one "
                             "(e.g. an appearance variant from scene_config.py).")
    parser.add_argument("--no-hsv-jitter", action="store_true",
                        help="Disable per-episode HSV jitter.")
    parser.add_argument("--ply", type=str, default=None,
                        help="Path to .ply or bare name under data/gsplat/. Default: "
                             "the --scene variant's splat, else wood.")
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="Render a splat that has no <ply_dir>/<name>/default.json.")
    args = parser.parse_args()

    # Import here (in the parent) only to use helpers — the heavy CUDA/EGL
    # state is created per-worker. Keep parent process light.
    import offline_renderer

    # Check the splat, its calibration and the scene's meshes before spawning.
    ply_path = offline_renderer.resolve_splat(args.ply, args.scene, args.allow_uncalibrated)
    offline_renderer.check_scene_assets(args.scene)
    enable_hsv_jitter = offline_renderer.ENABLE_HSV_JITTER and not args.no_hsv_jitter
    cameras = ("d435", "mid360") if args.mid360 else ("d435",)

    robot_files = offline_renderer._enumerate_episode_files(
        args.data_dir, episode=None, episode_from=args.episode_from,
    )
    if not robot_files:
        raise SystemExit(f"Error: No episode data found in {args.data_dir}")

    n_workers = max(1, min(args.workers, len(robot_files)))
    chunks = _split_round_robin(robot_files, n_workers)
    print(f"[MP] {len(robot_files)} episodes split across {n_workers} workers "
          f"(chunk sizes: {[len(c) for c in chunks]})")

    ctx = mp.get_context("spawn")
    gsplat_lock = ctx.Lock()
    procs = []
    t_start = time.perf_counter()
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = ctx.Process(
            target=_worker,
            args=(i, chunk, args.data_dir, ply_path,
                  args.scene, cameras, enable_hsv_jitter, gsplat_lock,
                  args.allow_uncalibrated),
            name=f"render-worker-{i}",
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
