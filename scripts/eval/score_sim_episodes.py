"""Score recorded sim rollouts: task success + stage-wise progress (App. B).

Reads the recorder's raw episodes (episode_XXXX.json, a list of per-frame
records, next to episode_XXXX_scenario.json) and scores each one from object,
plate and robot poses alone. Commanded gripper values are not used: in policy
rollouts they do not reflect the policy's gripper output.

Stages come from the scene's task_plan (scene_config.SCENES):
    approach/drive before the grasp -> walk   (base reaches the pick object)
    grasp                           -> pick   (object lifted and held at a wrist)
    heading after the grasp         -> turn   (held, facing the place target)
    approach/drive after the grasp  -> walk   (held, base reaches the place target)
    place                           -> place  (object resting on the target at the end)
which gives the paper's lists: locomani = walk, pick, place; turn = walk,
pick, turn, walk, place. Stages are cumulative: each one is searched for from
the frame where the previous one was met.

Usage:
    python scripts/eval/score_sim_episodes.py <episode_dir> [<episode_dir> ...] \
        [--scene orange_to_plate_turn] [--backbone psi0] [--condition "LEGS (200)"] \
        [--out results.csv]

--out writes one row per episode in the per-trial CSV format read by
plot_subtask_results.py (task, backbone, condition, trial, stages_completed,
stages, success, episode).
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main"))
from scene_config import get_scene  # noqa: E402

# All thresholds in one place (meters, radians, seconds).
TOLERANCES = {
    "walk_reach": 0.45,        # base xy to the walk target
    "lift_min": 0.03,          # pick: object z above its start
    "held_z_tol": 0.10,        # held: |object z - wrist z| at the nearest wrist
    "held_dist_max": 0.25,     # held: object to nearest wrist, 3D
    "hold_min_sec": 0.3,       # pick: held and lifted for this long
    "turn_tol": np.deg2rad(30.0),  # turn: heading vs bearing to the place target
    "place_xy_tol": 0.10,      # place: object xy to the target
    "place_dz_min": -0.05,     # place: object z - target z
    "place_dz_max": 0.06,
    "place_settle_sec": 0.3,   # place: on the target over the last 0.3 s
    "place_z_still_tol": 0.01,  # place: object z range within that window
}

TASK_NAMES = {
    "orange_to_plate_locomani": "locomani",
    "orange_to_plate_turn": "turn",
}


def base_scene(name):
    cfg = get_scene(name)
    return cfg.get("base_scene") or name


def plan_stages(task_plan):
    """[(stage, target), ...] from a scene task_plan."""
    pick_obj = next(t for k, t in task_plan if k == "grasp")
    place_tgt = next(t for k, t in task_plan if k == "place")
    stages, grasped = [], False
    for kind, _ in task_plan:
        if kind in ("approach", "drive"):
            if not (stages and stages[-1][0] == "walk"):
                stages.append(("walk", place_tgt if grasped else pick_obj))
        elif kind == "grasp":
            stages.append(("pick", pick_obj))
            grasped = True
        elif kind == "heading" and grasped:
            stages.append(("turn", place_tgt))
        elif kind == "place":
            stages.append(("place", place_tgt))
    return stages


def _quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_episode(path):
    """Per-frame arrays from one recorded episode."""
    with open(path) as f:
        frames = json.load(f)
    scen_path = path[:-len(".json")] + "_scenario.json"
    scenario = {}
    if os.path.exists(scen_path):
        with open(scen_path) as f:
            scenario = json.load(f)
    statics = {s["name"]: np.asarray(s["pos_w"], float) for s in scenario.get("statics", [])}

    t = np.array([fr["time_realtime_ms"] for fr in frames], float) / 1000.0
    base = np.array([fr["robot"]["base_pos_w"] for fr in frames], float)
    quats = [fr["robot"]["base_wxyz_w"] for fr in frames]
    rots = np.stack([_quat_wxyz_to_mat(q) for q in quats])
    yaw = np.arctan2(rots[:, 1, 0], rots[:, 0, 0])
    wrists = []
    for key in ("L_mat", "R_mat"):
        local = np.array([np.asarray(fr[key], float)[:3, 3] for fr in frames])
        wrists.append(np.einsum("nij,nj->ni", rots, local) + base)
    names = set().union(*(fr.get("objects", {}).keys() for fr in frames))
    objects = {}
    for name in names:
        pos = np.full((len(frames), 3), np.nan)
        for i, fr in enumerate(frames):
            if name in fr.get("objects", {}):
                pos[i] = fr["objects"][name]["pos_w"]
        objects[name] = pos
    return {"t": t - t[0], "base": base, "yaw": yaw, "wrists": np.stack(wrists, axis=1),
            "objects": objects, "statics": statics, "scenario": scenario}


def _entity_pos(ep, name):
    if name in ep["objects"]:
        return ep["objects"][name]
    if name in ep["statics"]:
        return np.repeat(ep["statics"][name][None], len(ep["t"]), axis=0)
    raise KeyError(f"'{name}' not in the episode's objects or statics")


def _first_sustained(mask, t, start, min_sec):
    """First index >= start where mask holds for min_sec, else None."""
    run_start = None
    for i in range(start, len(mask)):
        if mask[i]:
            if run_start is None:
                run_start = i
            if t[i] - t[run_start] >= min_sec:
                return run_start
        else:
            run_start = None
    return None


def _first(mask, start):
    idx = np.flatnonzero(mask[start:])
    return int(start + idx[0]) if len(idx) else None


def score_episode(ep, stages, tol=TOLERANCES):
    """Frame index at which each stage is met (None once one fails)."""
    t = ep["t"]
    pick_obj = next(tgt for s, tgt in stages if s == "pick")
    obj = _entity_pos(ep, pick_obj)
    wrist_d = np.linalg.norm(ep["wrists"] - obj[:, None, :], axis=2)  # (N, 2)
    nearest = np.argmin(wrist_d, axis=1)
    near_d = wrist_d[np.arange(len(t)), nearest]
    near_dz = np.abs(obj[:, 2] - ep["wrists"][np.arange(len(t)), nearest, 2])
    held = (near_d <= tol["held_dist_max"]) & (near_dz <= tol["held_z_tol"])
    lifted = obj[:, 2] - obj[0, 2] >= tol["lift_min"]

    frames, cursor = [], 0
    for stage, target in stages:
        tgt = _entity_pos(ep, target)
        if stage == "walk":
            dxy = np.linalg.norm(ep["base"][:, :2] - tgt[:, :2], axis=1)
            mask = dxy <= tol["walk_reach"]
            if target != pick_obj:
                mask &= held
            idx = _first(mask, cursor)
        elif stage == "pick":
            idx = _first_sustained(held & lifted, t, cursor, tol["hold_min_sec"])
        elif stage == "turn":
            d = tgt[:, :2] - ep["base"][:, :2]
            err = np.angle(np.exp(1j * (np.arctan2(d[:, 1], d[:, 0]) - ep["yaw"])))
            idx = _first(held & (np.abs(err) <= tol["turn_tol"]), cursor)
        elif stage == "place":
            dxy = np.linalg.norm(obj[:, :2] - tgt[:, :2], axis=1)
            dz = obj[:, 2] - tgt[:, 2]
            on = (dxy <= tol["place_xy_tol"]) & (dz >= tol["place_dz_min"]) & (dz <= tol["place_dz_max"])
            # Judged on the final frames: the object must end resting on the
            # target, i.e. on it at a constant height (a round object may still
            # roll on the plate). The wrist-to-object distance does not separate
            # held from released (both span ~0.13-0.2 m), so release is not checked.
            window = np.flatnonzero(t >= t[-1] - tol["place_settle_sec"])
            z_range = np.ptp(obj[window, 2])
            ok = (window[0] >= cursor and t[-1] - t[0] >= tol["place_settle_sec"]
                  and bool(on[window].all()) and z_range <= tol["place_z_still_tol"])
            idx = int(window[0]) if ok else None
        else:
            raise ValueError(f"Unknown stage '{stage}'")
        if idx is None:
            break
        frames.append(idx)
        cursor = idx
    return frames


def episode_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            cands = sorted(glob.glob(os.path.join(p, "episode_*.json")))
        else:
            cands = [p]
        files += [f for f in cands if not f.endswith(("_scenario.json", "_info.json"))]
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="Episode dirs or episode_XXXX.json files")
    ap.add_argument("--scene", default=None,
                    help="Scene name when the scenario file has none (recorded without --scene)")
    ap.add_argument("--backbone", default="", help="Value for the CSV's backbone column")
    ap.add_argument("--condition", default="", help="Value for the CSV's condition column")
    ap.add_argument("--out", default=None, help="Per-trial CSV to write")
    ap.add_argument("--verbose", action="store_true", help="Print every episode")
    args = ap.parse_args()

    files = episode_files(args.paths)
    if not files:
        sys.exit("No episode_XXXX.json files found")

    rows, totals = [], {}
    for trial, path in enumerate(files):
        ep = load_episode(path)
        scene = ep["scenario"].get("scene_name") or args.scene
        if not scene:
            sys.exit(f"{path}: scenario has no scene_name; pass --scene")
        base = base_scene(scene)
        stages = plan_stages(get_scene(scene)["task_plan"])
        frames = score_episode(ep, stages)
        names = [s for s, _ in stages]
        n = len(frames)
        row = {"task": TASK_NAMES.get(base, base), "backbone": args.backbone,
               "condition": args.condition, "trial": trial, "stages_completed": n,
               "stages": "|".join(names), "success": int(n == len(stages)),
               "episode": os.path.relpath(path)}
        rows.append(row)
        key = (row["task"], row["stages"])
        totals.setdefault(key, []).append(n)
        if args.verbose:
            reached = ", ".join(f"{s}@{ep['t'][i]:.1f}s" for s, i in zip(names, frames))
            print(f"{row['episode']}: {n}/{len(stages)} [{reached}]")

    for (task, stage_str), counts in totals.items():
        names = stage_str.split("|")
        counts = np.asarray(counts)
        cum = " ".join(f"{s} {100 * np.mean(counts >= k + 1):.0f}%" for k, s in enumerate(names))
        print(f"{task}: {np.sum(counts == len(names))}/{len(counts)} success | {cum}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.out} ({len(rows)} trials)")


if __name__ == "__main__":
    main()
