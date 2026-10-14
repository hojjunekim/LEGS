r"""
Stage-wise cumulative success plot (paper App. B, Fig. 6).

Reads per-trial CSVs and draws one panel per (backbone, task): the percentage
of trials still successful after each subtask stage, one line per data
condition. The last stage equals the end-task success rate (TSR). A summary
table is printed as well.

Per-trial CSV, one row per trial (score_sim_episodes.py --out writes this):
    task              locomani | turn, or any name if `stages` is given
    backbone          psi0 | pi05 | gr00t (other labels are plotted after these)
    condition         e.g. "LEGS (200)", "LEGS (50)", "SAM3D (200)", "Teleop (50)"
    trial             trial index within its (task, backbone, condition) cell
    stages_completed  stages completed in order, 0 .. number of stages
    stages            optional: stage names joined by "|" (default: the paper's, below)
    success           optional: 1 if every stage was completed (checked)
    episode           optional: the recording the trial was scored from

Paper stages: locomani = walk, pick, place;
turn = walk, pick, turn, walk, place (squat).

Usage:
    python scripts/eval/plot_subtask_results.py results.csv [more.csv ...] \
        [--output stagewise_success.pdf]

Use .pdf for the paper (vector) or .png for previews.
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


REQUIRED = ("task", "backbone", "condition", "trial", "stages_completed")

STAGES = {
    "locomani": ["walk", "pick", "place"],
    "turn": ["walk", "pick", "turn", "walk", "place\n(squat)"],
}
TASK_TITLES = {
    "locomani": "Task 2: Loco-manipulation",
    "turn": "Task 3: Long-horizon",
}
BACKBONE_TITLES = {"psi0": r"$\psi_0$", "pi05": r"$\pi_{0.5}$", "gr00t": "GR00T N1.6"}
CONDITION_STYLES = {
    "LEGS (200)": dict(color="#c0392b", marker="o", linewidth=2.6, markersize=6.5,
                       label="LEGS (200) [ours]"),
    "LEGS (50)": dict(color="#e67e22", marker="D", linewidth=1.5, markersize=5),
    "SAM3D (200)": dict(color="#2e6fb5", marker="s", linewidth=1.5, markersize=5),
    "Teleop (50)": dict(color="#7f7f7f", marker="^", linewidth=1.5, markersize=5.5),
}
Y_OFFSET = 1.2  # percentage points between conditions, keeps overlapping lines visible


def ordered(keys, preferred):
    return [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)


def load_trials(paths):
    cells = defaultdict(list)  # (task, backbone, condition) -> rows
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                raise SystemExit(f"{path}: missing columns {missing}")
            for row in reader:
                cells[(row["task"], row["backbone"], row["condition"])].append(row)
    if not cells:
        raise SystemExit("No trials found")
    return cells


def stage_labels(task, rows):
    """Display labels for a cell's stages; checks every row agrees."""
    given = {r.get("stages") or "" for r in rows}
    if len(given) > 1:
        raise SystemExit(f"Task '{task}': trials disagree on stages: {sorted(given)}")
    names = given.pop()
    n = len(names.split("|")) if names else None
    if task in STAGES and n in (None, len(STAGES[task])):
        return STAGES[task]
    if names:
        return names.split("|")
    raise SystemExit(f"Task '{task}' has no default stages; add a 'stages' column")


def cumulative_success(rows, n_stages):
    """Percent of trials with at least k + 1 stages completed, k = 0 .. n_stages - 1."""
    done = np.array([int(r["stages_completed"]) for r in rows])
    if done.min() < 0 or done.max() > n_stages:
        raise SystemExit(f"stages_completed outside 0..{n_stages}: {sorted(set(done))}")
    for r, d in zip(rows, done):
        if r.get("success") not in (None, "") and int(r["success"]) != int(d == n_stages):
            raise SystemExit(f"Trial {r['trial']}: success={r['success']} but "
                             f"stages_completed={d} of {n_stages}")
    return np.array([100.0 * np.mean(done >= k + 1) for k in range(n_stages)])


def plot(cells, output):
    tasks = ordered({k[0] for k in cells}, list(STAGES))
    backbones = ordered({k[1] for k in cells}, list(BACKBONE_TITLES))
    conditions = ordered({k[2] for k in cells}, list(CONDITION_STYLES))
    labels = {t: stage_labels(t, [r for k, rows in cells.items() if k[0] == t for r in rows])
              for t in tasks}
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = {c: CONDITION_STYLES.get(c, dict(color=fallback[i % len(fallback)], marker="v",
                                              linewidth=1.5, markersize=5))
              for i, c in enumerate(conditions)}
    offsets = {c: (i - (len(conditions) - 1) / 2) * Y_OFFSET for i, c in enumerate(conditions)}

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    fig, axes = plt.subplots(len(backbones), len(tasks), squeeze=False,
                             figsize=(3.0 * len(tasks) + 1.5, 2.3 * len(backbones) + 0.6),
                             gridspec_kw={"width_ratios": [len(labels[t]) + 1 for t in tasks]})
    print(f"{'task':<10} {'backbone':<9} {'condition':<14} {'n':>3}  cumulative success per stage")
    for i, backbone in enumerate(backbones):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            x = np.arange(len(labels[task]))
            for cond in conditions:
                rows = cells.get((task, backbone, cond))
                if not rows:
                    continue
                pct = cumulative_success(rows, len(x))
                style = {k: v for k, v in styles[cond].items() if k != "label"}
                ax.plot(x, pct + offsets[cond], zorder=3, clip_on=False, **style)
                stages = ", ".join(f"{s.split(chr(10))[0]} {p:.0f}%" for s, p in zip(labels[task], pct))
                print(f"{task:<10} {backbone:<9} {cond:<14} {len(rows):>3}  {stages}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels[task], fontsize=9)
            ax.set_xlim(-0.4, len(x) - 0.6)
            ax.set_ylim(-5, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.tick_params(labelsize=9)
            ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, linestyle="--")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(TASK_TITLES.get(task, task), fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{BACKBONE_TITLES.get(backbone, backbone)}\nsuccess (%)", fontsize=10)
            else:
                ax.set_yticklabels([])

    handles = [Line2D([], [], **{k: v for k, v in styles[c].items() if k != "label"})
               for c in conditions]
    names = [styles[c].get("label", c) for c in conditions]
    fig.legend(handles, names, loc="lower center", ncol=len(conditions), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.35 / fig.get_figheight(), 1, 1))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", help="Per-trial CSV file(s)")
    parser.add_argument("--output", default="stagewise_success.png",
                        help="Figure path (.pdf for vector, .png for previews)")
    args = parser.parse_args()
    plot(load_trials(args.csv), args.output)


if __name__ == "__main__":
    main()
