"""
Open-loop evaluation of a LEGS policy server on LeRobot dataset episodes.

Replays recorded episodes through a running psi0, pi05 or gr00t server (see
scripts/main/utils/psi0_client.py), compares every predicted chunk with the
recorded actions, and saves per-dimension errors as CSVs. Each query is sent
with reset=True, so samples are independent (psi0 keeps no RTC chunk between
them). Render figures with openloop_plot.py.

Usage:
    python scripts/eval/openloop_eval.py --backend psi0 \
        --dataset-path <lerobot_dataset> --episode 0-9

Outputs (per episode N) under {save_dir}/{dataset_name}/{backend}/:
    ep{N}_meta.json        episode metadata (frames, stride, horizon, labels, units)
    ep{N}_gt.csv           recorded actions (frame x action dims, raw units)
    ep{N}_first_step.csv   first-step L1 error per sample (raw units)
    ep{N}_horizon_avg.csv  horizon-averaged L1 error per sample (raw units)
    ep{N}_chunks.csv       predicted chunks in long format (raw units)
    ep{N}_timings.csv      per-sample inference time (ms)
    summary.json           mean per-dim errors of this run (angles in degrees)

Errors on the wrist roll/pitch/yaw dims are wrapped to [-pi, pi]. The dataset's
parquet and mp4 files are read directly (LeRobot v2.1 layout), so lerobot itself
is not needed. Requires numpy, pandas, pyarrow, av and requests (plus
openpi_client and websockets for the pi05 backend).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main"))
from utils.psi0_client import BACKENDS, make_client  # noqa: E402

# 18-D LEGS action (action.psi0_18): L wrist xyz + rpy, L grip, R wrist xyz + rpy,
# R grip, base vx, vy, vyaw command, height.
ACTION_LABELS = [
    "L_x(m)", "L_y(m)", "L_z(m)", "L_roll(deg)", "L_pitch(deg)", "L_yaw(deg)", "L_grip",
    "R_x(m)", "R_y(m)", "R_z(m)", "R_roll(deg)", "R_pitch(deg)", "R_yaw(deg)", "R_grip",
    "vx(m/s)", "vy(m/s)", "vyaw(deg/s)", "height(m)",
]
WRAP_DIMS = [3, 4, 5, 10, 11, 12]  # Euler angles: errors wrapped to [-pi, pi]
ANGLE_DIMS = WRAP_DIMS + [16]      # radians in the data, degrees in reports


class LeRobotEpisodes:
    """Per-episode parquet + mp4 access to a LeRobot v2.1 dataset."""

    def __init__(self, root):
        self.root = Path(root).expanduser()
        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)
        self.tasks = {}
        tasks_path = self.root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.tasks[row["task_index"]] = row["task"]

    @property
    def total_episodes(self):
        return self.info["total_episodes"]

    def _path(self, template, ep, **fmt):
        chunk = ep // self.info["chunks_size"]
        return self.root / self.info[template].format(episode_chunk=chunk, episode_index=ep, **fmt)

    def table(self, ep):
        return pd.read_parquet(self._path("data_path", ep))

    def frames(self, ep, video_key):
        """(H, W, 3) uint8 RGB frames of one episode, in order."""
        import av
        with av.open(str(self._path("video_path", ep, video_key=video_key))) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")


def column(df, key):
    if key not in df.columns:
        raise KeyError(f"'{key}' is not a dataset column; available: {list(df.columns)}")
    return np.stack(df[key].to_numpy()).astype(np.float32)


def action_error(pred, gt):
    """|pred - gt| per step and dim, Euler angle errors wrapped to [-pi, pi]."""
    err = np.abs(pred - gt)
    wrap = [d for d in WRAP_DIMS if d < err.shape[1]]
    err[:, wrap] = np.abs(np.angle(np.exp(1j * (pred[:, wrap] - gt[:, wrap]))))
    return err


def to_display(arr):
    """Angle dims (last axis) from radians to degrees."""
    out = np.array(arr, dtype=np.float64)
    dims = [d for d in ANGLE_DIMS if d < out.shape[-1]]
    out[..., dims] = np.rad2deg(out[..., dims])
    return out


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def eval_episode(ds, client, ep, args, horizon, save_dir):
    df = ds.table(ep)
    states = column(df, args.state_key)
    gt = column(df, args.action_key)
    n_frames, n_dims = gt.shape
    prompt = args.prompt
    if prompt is None:
        prompt = ds.tasks.get(int(df["task_index"].iloc[0]), "") if "task_index" in df else ""
    print(f"\n--- Episode {ep}: {n_frames} frames, prompt: {prompt!r} ---")

    first_rows, horizon_rows, chunk_rows, timing_rows = [], [], [], []
    n_video = 0
    for i, image in enumerate(ds.frames(ep, args.image_key)):
        n_video += 1
        if i >= n_frames or i % args.stride:
            continue
        t0 = time.perf_counter()
        pred = client.infer({"image": image, "state": states[i], "prompt": prompt, "reset": True})["actions"]
        timing_rows.append([len(timing_rows), 1000.0 * (time.perf_counter() - t0)])
        if pred.ndim != 2 or pred.shape[1] != n_dims:
            raise ValueError(f"Expected (T, {n_dims}) actions from the server, got {pred.shape}")
        pred = pred[:horizon]
        steps = min(len(pred), n_frames - i)
        err = action_error(pred[:steps], gt[i:i + steps])
        sample = len(first_rows)
        first_rows.append([sample, i] + err[0].tolist())
        horizon_rows.append([sample, i] + err.mean(axis=0).tolist())
        chunk_rows += [[sample, i, k, i + k] + pred[k].tolist() for k in range(len(pred))]
        if sample % 25 == 0:
            print(f"  frame {i}/{n_frames}  {timing_rows[-1][1]:.0f} ms", flush=True)
    if n_video != n_frames:
        print(f"  [warn] video has {n_video} frames, parquet has {n_frames}")
    if not first_rows:
        raise RuntimeError(f"Episode {ep}: no frames decoded from {args.image_key}")

    dim_header = [f"dim_{d}" for d in range(n_dims)]
    write_csv(save_dir / f"ep{ep}_gt.csv", ["frame"] + dim_header,
              [[t] + gt[t].tolist() for t in range(n_frames)])
    write_csv(save_dir / f"ep{ep}_first_step.csv", ["sample", "frame"] + dim_header, first_rows)
    write_csv(save_dir / f"ep{ep}_horizon_avg.csv", ["sample", "frame"] + dim_header, horizon_rows)
    write_csv(save_dir / f"ep{ep}_chunks.csv",
              ["sample", "frame_offset", "step_in_chunk", "frame"] + dim_header, chunk_rows)
    write_csv(save_dir / f"ep{ep}_timings.csv", ["sample", "time_ms"], timing_rows)

    meta = {
        "episode": int(ep),
        "ep_len": int(n_frames),
        "stride": int(args.stride),
        "action_horizon": int(horizon),
        "n_action_dim": int(n_dims),
        "action_labels": ACTION_LABELS[:n_dims],
        "angle_dims": [d for d in ANGLE_DIMS if d < n_dims],
        "wrap_dims": [d for d in WRAP_DIMS if d < n_dims],
        "n_samples": len(first_rows),
        "mean_inference_time_ms": float(np.mean([r[1] for r in timing_rows])),
        "backend": args.backend,
        "prompt": prompt,
    }
    with open(save_dir / f"ep{ep}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {meta['n_samples']} samples, mean inference {meta['mean_inference_time_ms']:.1f} ms")
    first = np.array([r[2:] for r in first_rows])
    horizon_avg = np.array([r[2:] for r in horizon_rows])
    return first, horizon_avg


def parse_episodes(spec, n_episodes):
    """'5', '0-9' (inclusive) or '0,5,9'."""
    eps = []
    for part in spec.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            eps += list(range(int(start), int(end) + 1))
        else:
            eps.append(int(part))
    bad = [e for e in eps if not 0 <= e < n_episodes]
    if bad:
        raise SystemExit(f"Episodes {bad} out of range (dataset has {n_episodes})")
    return eps


def print_table(title, n_samples, first, horizon_avg, labels):
    print(f"\n{title}: mean L1 over {n_samples} samples (angles in degrees)")
    print(f"  {'dim':>3}  {'label':<12} {'first-step':>11} {'horizon-avg':>12}")
    for d, label in enumerate(labels):
        print(f"  {d:>3}  {label:<12} {first[d]:>11.4f} {horizon_avg[d]:>12.4f}")


def main():
    ports = ", ".join(f"{k} {v['port']}" for k, v in BACKENDS.items())
    chunks = ", ".join(f"{k} {v['chunk']}" for k, v in BACKENDS.items())
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=list(BACKENDS), default="psi0")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=None,
                        help=f"Server port (default per backend: {ports})")
    parser.add_argument("--dataset-path", required=True,
                        help="LeRobot v2.1 dataset root (the dir holding meta/info.json)")
    parser.add_argument("--episode", default="0",
                        help="Episode index, inclusive range '0-9' or list '0,5,9' (default 0)")
    parser.add_argument("--stride", type=int, default=4, help="Query every N-th frame")
    parser.add_argument("--horizon", type=int, default=None,
                        help=f"Chunk steps scored (default: the executed chunk, {chunks})")
    parser.add_argument("--image-key", default="observation.images.ego_view")
    parser.add_argument("--state-key", default="observation.state_psi0_18")
    parser.add_argument("--action-key", default="action.psi0_18")
    parser.add_argument("--prompt", default=None,
                        help="Instruction sent to the policy (default: the episode's task)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-query timeout (s)")
    parser.add_argument("--save-dir", default=str(Path(__file__).parent / "eval_results"))
    args = parser.parse_args()

    horizon = args.horizon or BACKENDS[args.backend]["chunk"]
    ds = LeRobotEpisodes(args.dataset_path)
    episodes = parse_episodes(args.episode, ds.total_episodes)
    client = make_client(args.backend, host=args.host, port=args.port, timeout=args.timeout)
    if not getattr(client, "connected", True):
        raise SystemExit(f"No {args.backend} server reachable at {args.host}; start it first")

    save_dir = Path(args.save_dir) / ds.root.name / args.backend
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.backend}: episodes {episodes}, stride {args.stride}, horizon {horizon}")

    per_ep, all_first, all_horizon = {}, [], []
    for ep in episodes:
        first, horizon_avg = eval_episode(ds, client, ep, args, horizon, save_dir)
        per_ep[ep] = {"n_samples": len(first),
                      "first_step": to_display(first).mean(axis=0).tolist(),
                      "horizon_avg": to_display(horizon_avg).mean(axis=0).tolist()}
        all_first.append(first)
        all_horizon.append(horizon_avg)

    first = to_display(np.concatenate(all_first)).mean(axis=0)
    horizon_avg = to_display(np.concatenate(all_horizon)).mean(axis=0)
    labels = ACTION_LABELS[:len(first)]
    print_table(f"{args.backend} on {ds.root.name}, episodes {args.episode}",
                sum(v["n_samples"] for v in per_ep.values()), first, horizon_avg, labels)
    summary = {"backend": args.backend, "dataset": ds.root.name, "stride": args.stride,
               "action_horizon": horizon, "action_labels": labels,
               "overall": {"first_step": first.tolist(), "horizon_avg": horizon_avg.tolist()},
               "episodes": per_ep}
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {save_dir}")
    print(f"Plot with: python scripts/eval/openloop_plot.py --save-dir {save_dir}")


if __name__ == "__main__":
    main()
