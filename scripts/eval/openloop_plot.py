"""
Plot open-loop eval results from the CSVs written by openloop_eval.py.

Usage:
    python scripts/eval/openloop_plot.py --save-dir <eval_dir> [--episode 0-9]

<eval_dir> is the per-backend directory written by the eval script, e.g.
scripts/eval/eval_results/<dataset_name>/<backend>. Without --episode, every
episode found there is plotted.

Outputs under <eval_dir>:
    l1_per_dim_ep{N}.png    mean first-step / horizon-avg error per action dim
    l1_over_time_ep{N}.png  first-step error over the episode, by dim group
    gt_vs_pred_ep{N}.png    recorded actions vs every predicted chunk
    l1_per_dim_all.png      per-dim error pooled over the plotted episodes

Only reads CSVs, so figures can be restyled without re-running the policy.
Requires numpy and matplotlib.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# Fallbacks if meta.json is incomplete (same as openloop_eval.py).
ACTION_LABELS_DEFAULT = [
    "L_x(m)", "L_y(m)", "L_z(m)", "L_roll(deg)", "L_pitch(deg)", "L_yaw(deg)", "L_grip",
    "R_x(m)", "R_y(m)", "R_z(m)", "R_roll(deg)", "R_pitch(deg)", "R_yaw(deg)", "R_grip",
    "vx(m/s)", "vy(m/s)", "vyaw(deg/s)", "height(m)",
]
ANGLE_DIMS_DEFAULT = [3, 4, 5, 10, 11, 12, 16]
DIM_GROUPS = [
    ("Left wrist xyz", [0, 1, 2], "m"),
    ("Left wrist rpy", [3, 4, 5], "deg"),
    ("Right wrist xyz", [7, 8, 9], "m"),
    ("Right wrist rpy", [10, 11, 12], "deg"),
    ("Grippers (0.5 open, 1.0 closed)", [6, 13], ""),
    ("Base velocity (vx, vy)", [14, 15], "m/s"),
    ("Base yaw rate (vyaw)", [16], "deg/s"),
    ("Height", [17], "m"),
]


def to_display(arr, angle_dims):
    """Angle dims from radians to degrees."""
    out = arr.copy()
    for d in angle_dims:
        if d < out.shape[1]:
            out[:, d] = np.rad2deg(out[:, d])
    return out


def groups_for(n_dims):
    groups = [(t, [d for d in dims if d < n_dims], u) for t, dims, u in DIM_GROUPS]
    return [g for g in groups if g[1]]


def read_csv_floats(path, n_skip_cols):
    """(header, leading int columns, trailing float array) of a CSV."""
    with open(path) as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    if not body:
        return header, np.zeros((0, n_skip_cols), dtype=int), np.zeros((0, max(0, len(header) - n_skip_cols)))
    leading = np.array([[int(x) for x in r[:n_skip_cols]] for r in body])
    data = np.array([[float(x) for x in r[n_skip_cols:]] for r in body])
    return header, leading, data


def reconstruct_chunks(ch_lead, ch_data):
    """Long-format chunk rows back into [(frame_offset, chunk_array), ...]."""
    chunks = []
    for sample_idx in np.unique(ch_lead[:, 0]):
        mask = ch_lead[:, 0] == sample_idx
        order = np.argsort(ch_lead[mask, 2])
        chunks.append((int(ch_lead[mask, 1][0]), ch_data[mask][order]))
    chunks.sort(key=lambda c: c[0])
    return chunks


def plot_l1_per_dim(path, title, mean_first, mean_horizon, horizon, action_labels):
    n_dims = len(mean_first)
    x = np.arange(n_dims)
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar(x - width / 2, mean_first, width, label="First step", color="steelblue")
    bars2 = ax.bar(x + width / 2, mean_horizon, width,
                   label=f"Horizon avg ({horizon} steps)", color="coral")
    ax.set_ylabel("Mean L1 error (m, deg, m/s, deg/s)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(action_labels[:n_dims], rotation=45, ha="right")
    ax.legend()
    for bars, vals in ((bars1, mean_first), (bars2, mean_horizon)):
        for bar, val in zip(bars, vals):
            if val > 0.0001:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}",
                        ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close("all")


def plot_l1_over_time(save_dir, ep_idx, first_step_errors, horizon_errors, frames,
                      stride, action_labels):
    groups = groups_for(first_step_errors.shape[1])
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 2.8 * len(groups)), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"Open-loop first-step error over time | Episode {ep_idx} | "
                 f"mean horizon-avg L1 {np.mean(horizon_errors):.4f}", fontsize=13)
    for ax, (title, dims, unit) in zip(axes, groups):
        for d in dims:
            ax.plot(frames, first_step_errors[:, d], label=action_labels[d])
        ax.set_title(title)
        ax.set_ylabel(f"L1 error ({unit})" if unit else "L1 error")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(f"Frame (one sample every {stride})")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))  # room for the suptitle
    path = save_dir / f"l1_over_time_ep{ep_idx}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close("all")


def plot_gt_vs_pred(save_dir, ep_idx, gt_disp, pred_chunks, ep_len, angle_dims, action_labels):
    groups = groups_for(gt_disp.shape[1])
    tab10 = plt.cm.tab10
    colors = {d: tab10(i % 10) for i, d in enumerate(d for _, dims, _ in groups for d in dims)}
    fig, axes = plt.subplots(len(groups), 1, figsize=(16, 4.0 * len(groups)), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"Recorded vs predicted chunks | Episode {ep_idx}", fontsize=24, fontweight="bold")
    for ax, (title, dims, unit) in zip(axes, groups):
        for d in dims:
            ax.plot(np.arange(ep_len), gt_disp[:, d], "-", color=colors[d],
                    label=f"GT {action_labels[d]}", alpha=0.95, linewidth=2.6)
        for chunk_idx, (frame_offset, chunk) in enumerate(pred_chunks):
            chunk_disp = to_display(chunk, angle_dims)
            x_chunk = np.arange(frame_offset, frame_offset + len(chunk_disp))
            for d in dims:
                ax.plot(x_chunk, chunk_disp[:, d], "--", color=colors[d], alpha=0.75, linewidth=2.2,
                        label=f"Pred {action_labels[d]}" if chunk_idx == 0 else None)
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_ylabel(f"Value ({unit})" if unit else "Value", fontsize=17)
        ax.tick_params(axis="both", labelsize=15)
        ax.legend(fontsize=15, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame (within episode)", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
    path = save_dir / f"gt_vs_pred_ep{ep_idx}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close("all")


def plot_episode(save_dir, ep_idx):
    """Per-episode figures; returns (first-step errors, horizon-avg errors, meta), angles in degrees."""
    meta_path = save_dir / f"ep{ep_idx}_meta.json"
    if not meta_path.exists():
        print(f"[skip] ep{ep_idx}: no {meta_path.name}")
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    n_dims = meta["n_action_dim"]
    labels = meta.get("action_labels", ACTION_LABELS_DEFAULT[:n_dims])
    angle_dims = meta.get("angle_dims", [d for d in ANGLE_DIMS_DEFAULT if d < n_dims])

    _, _, gt_full = read_csv_floats(save_dir / f"ep{ep_idx}_gt.csv", n_skip_cols=1)
    _, first_lead, l1_first = read_csv_floats(save_dir / f"ep{ep_idx}_first_step.csv", n_skip_cols=2)
    _, _, l1_horizon = read_csv_floats(save_dir / f"ep{ep_idx}_horizon_avg.csv", n_skip_cols=2)
    _, ch_lead, ch_data = read_csv_floats(save_dir / f"ep{ep_idx}_chunks.csv", n_skip_cols=4)
    _, _, timings = read_csv_floats(save_dir / f"ep{ep_idx}_timings.csv", n_skip_cols=1)

    first_disp = to_display(l1_first, angle_dims)
    horizon_disp = to_display(l1_horizon, angle_dims)
    mean_time = float(np.mean(timings)) if timings.size else 0.0
    backend = meta.get("backend", "")
    plot_l1_per_dim(save_dir / f"l1_per_dim_ep{ep_idx}.png",
                    f"Open-loop L1 error | {backend} | Episode {ep_idx} | inference {mean_time:.0f} ms",
                    first_disp.mean(axis=0), horizon_disp.mean(axis=0), meta["action_horizon"], labels)
    plot_l1_over_time(save_dir, ep_idx, first_disp, horizon_disp, first_lead[:, 1],
                      meta["stride"], labels)
    plot_gt_vs_pred(save_dir, ep_idx, to_display(gt_full, angle_dims),
                    reconstruct_chunks(ch_lead, ch_data), meta["ep_len"], angle_dims, labels)
    return first_disp, horizon_disp, meta


def parse_episode_arg(episode_str):
    if episode_str is None or episode_str == "all":
        return None
    eps = []
    for part in episode_str.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            eps += list(range(int(start), int(end) + 1))
        else:
            eps.append(int(part))
    return eps


def discover_episodes(save_dir):
    eps = []
    for p in save_dir.glob("ep*_meta.json"):
        try:
            eps.append(int(p.stem.split("_")[0][2:]))
        except ValueError:
            continue
    return sorted(eps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--save-dir", required=True,
                        help="Per-backend dir written by openloop_eval.py")
    parser.add_argument("--episode", default=None,
                        help="Episode index, range '0-9', list '0,5,9' or 'all' (default: all found)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    if not save_dir.is_dir():
        raise SystemExit(f"save_dir not found: {save_dir}")
    ep_indices = parse_episode_arg(args.episode) or discover_episodes(save_dir)
    if not ep_indices:
        raise SystemExit(f"No ep*_meta.json found in {save_dir}")

    print(f"Plotting episodes: {ep_indices}")
    results = [r for r in (plot_episode(save_dir, ep) for ep in ep_indices) if r is not None]
    if len(results) > 1:
        first = np.concatenate([r[0] for r in results])
        horizon = np.concatenate([r[1] for r in results])
        meta = results[0][2]
        plot_l1_per_dim(save_dir / "l1_per_dim_all.png",
                        f"Open-loop L1 error | {meta.get('backend', '')} | "
                        f"{len(results)} episodes, {len(first)} samples",
                        first.mean(axis=0), horizon.mean(axis=0), meta["action_horizon"],
                        meta.get("action_labels", ACTION_LABELS_DEFAULT))
    print(f"\nDone. Figures saved to {save_dir}")


if __name__ == "__main__":
    main()
