"""Augment a GR00T-WBC teleop LeRobot dataset with Psi0 columns in place.

Appends two columns to every parquet under <root>/data/chunk-*/episode_*.parquet:
    observation.state_psi0   (float32, [15]) = L_eef_6d + L_grip + R_eef_6d + R_grip + height
    action.psi0_18           (float32, [18]) = L_eef_6d + L_grip + R_eef_6d + R_grip + vx + vy + vyaw + height

Computed entirely from existing GR00T-WBC columns:
    observation.eef_state         (14,) double — pelvis-relative wrists, wxyz quat
    observation.root_orientation  (4,)  double — base wxyz quat; vyaw = ddt(yaw) * FPS
    teleop.{left,right}_hand_joints (7,) float — finger joints; grip = signed encoding (see _hand_to_grip)
    teleop.planner_movement       (3,)  float — vx, vy, vz (we use [:2])
    teleop.planner_mode           int           — 0=idle (vx, vy zeroed; vyaw kept from root_orientation)
    teleop.planner_height         float         — sentinel -1 → --height fallback

Grip and vyaw encodings are aligned with the rollout side
(scripts/main/utils/policy_runner.py::_interpolate_hand and
scripts/main/simulate_object.py's gripper_map): lego=0.5, stretch=0.0, grab=1.0
so the model's output convention is consistent end-to-end. vyaw is reconstructed
from the actual recorded base orientation: the teleop columns hold no yaw-rate
command, but the resulting body rotation is in observation.root_orientation.

Also updates meta/info.json with the two new feature entries. Does NOT write
modality_psi0.json: psi0 does not read it (scripts/train/psi0.sh passes the
column keys on the command line). GR00T fine-tune keeps using the existing modality.json.

Next steps (mirrors wigs path):
    .venv-openpi/bin/python scripts/data/add_18state_column.py --root <root>
    .venv-openpi/bin/python submodules/Psi0/scripts/data/calc_modality_stats.py --task-dir <root>

Usage:
    .venv-openpi/bin/python scripts/data/add_psi0_columns_teleop.py \\
        --root data/lerobot/teleop/glue_to_basket [--height 0.78]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

FPS = 30


def _quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def _pose_to_6d(pos, quat_xyzw):
    rpy = R.from_quat(quat_xyzw).as_euler("xyz")
    return np.concatenate([pos, rpy]).astype(np.float32)


def _hand_to_grip(hand_joints_7, is_left_hand):
    """7-DOF hand joints → two-mode {0.5, 1.0} step signal.

    Inverts ``stretch * (1 - t) + grab * t`` for indices 3..6 (same formula
    as before — ``stretch[3:7]`` = ``+1.5`` left / ``-1.5`` right), then
    snaps the continuous estimate to a discrete grip command:
        cont >= 0.75  → 1.0  (grab)
        cont <  0.75  → 0.5  (idle / stretch both collapse to neutral)

    Two-mode by design — matches the universal uni_20/uni_23 schema so
    teleop and wigs grips share one distribution. The previous mean-based
    [0,1] continuous output was effectively two clusters anyway (~0.5,
    ~0.917); snapping makes the cluster boundaries explicit.
    """
    sign = -1.0 if is_left_hand else 1.0
    cont = 0.5 + sign * np.mean(hand_joints_7[3:7]) / 3.0
    return 1.0 if cont >= 0.75 else 0.5


def _yaw_from_quat_wxyz(quat_wxyz_arr):
    """(N, 4) wxyz quat → (N,) unwrapped yaw in radians."""
    xyzw = quat_wxyz_arr[:, [1, 2, 3, 0]]
    yaw = R.from_quat(xyzw).as_euler("xyz")[:, 2]
    return np.unwrap(yaw)


def _extract_base_cmd(table, height_const):
    n = table.num_rows
    cols = set(table.column_names)
    has_planner = {"teleop.planner_movement", "teleop.planner_mode"} <= cols

    if has_planner:
        mov  = np.array(table.column("teleop.planner_movement").to_pylist(), dtype=np.float32)
        mode = np.array(table.column("teleop.planner_mode").to_pylist(),     dtype=np.int32).reshape(-1)
        idle = mode == 0
        vxy = mov[:, :2].copy()
        vxy[idle] = 0.0
    else:
        vxy = np.zeros((n, 2), dtype=np.float32)

    # vyaw from FD of the actual recorded base orientation. The teleop
    # columns hold no yaw-rate command, and the legacy
    # gradient-of-planner_facing approach yields zeros because facing is
    # constant within each non-idle segment. The robot's true yaw rate lives in
    # observation.root_orientation regardless of planner mode.
    if "observation.root_orientation" in cols:
        ro = np.array(table.column("observation.root_orientation").to_pylist(), dtype=np.float64)
        yaw_unwrapped = _yaw_from_quat_wxyz(ro)
        vyaw = (np.gradient(yaw_unwrapped) * FPS).astype(np.float32)
    else:
        vyaw = np.zeros(n, dtype=np.float32)

    if "teleop.planner_height" in cols:
        hgt = np.array(table.column("teleop.planner_height").to_pylist(), dtype=np.float32).reshape(-1)
        height = np.where(hgt > -0.5, hgt, height_const).astype(np.float32)
    else:
        height = np.full(n, height_const, dtype=np.float32)

    return vxy, vyaw, height


def augment_parquet(path: Path, height_const: float) -> bool:
    table = pq.read_table(path)
    if {"observation.state_psi0", "action.psi0_18"} <= set(table.column_names):
        return False

    n = table.num_rows
    eef = np.array(table.column("observation.eef_state").to_pylist(), dtype=np.float32)
    lh  = np.array(table.column("teleop.left_hand_joints").to_pylist(),  dtype=np.float32)
    rh  = np.array(table.column("teleop.right_hand_joints").to_pylist(), dtype=np.float32)
    vxy, vyaw, height = _extract_base_cmd(table, height_const)

    state_15  = np.empty((n, 15), dtype=np.float32)
    action_18 = np.empty((n, 18), dtype=np.float32)
    for t in range(n):
        l_6d = _pose_to_6d(eef[t, 0:3],  _quat_wxyz_to_xyzw(eef[t, 3:7]))
        r_6d = _pose_to_6d(eef[t, 7:10], _quat_wxyz_to_xyzw(eef[t, 10:14]))
        l_grip = _hand_to_grip(lh[t], is_left_hand=True)
        r_grip = _hand_to_grip(rh[t], is_left_hand=False)
        state_15[t]      = np.concatenate([l_6d, [l_grip], r_6d, [r_grip], [height[t]]])
        action_18[t, :14] = state_15[t, :14]
        action_18[t, 14:] = [vxy[t, 0], vxy[t, 1], vyaw[t], height[t]]

    table = table.append_column("observation.state_psi0",
                                pa.array(state_15.tolist(),  type=pa.list_(pa.float32(), 15)))
    table = table.append_column("action.psi0_18",
                                pa.array(action_18.tolist(), type=pa.list_(pa.float32(), 18)))

    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return True


def update_info_json(root: Path) -> None:
    p = root / "meta/info.json"
    with open(p) as f:
        info = json.load(f)
    feats = info["features"]

    state_names = [
        "L_eef_x", "L_eef_y", "L_eef_z", "L_eef_roll", "L_eef_pitch", "L_eef_yaw", "L_grip",
        "R_eef_x", "R_eef_y", "R_eef_z", "R_eef_roll", "R_eef_pitch", "R_eef_yaw", "R_grip",
        "height",
    ]
    action_names = state_names[:14] + ["vx", "vy", "vyaw", "height"]
    feats["observation.state_psi0"] = {"dtype": "float32", "shape": [15], "names": state_names}
    feats["action.psi0_18"]         = {"dtype": "float32", "shape": [18], "names": action_names}

    with open(p, "w") as f:
        json.dump(info, f, indent=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Teleop LeRobot dataset root (modified in place)")
    ap.add_argument("--height", type=float, default=0.78,
                    help="Constant base-height when planner_height==-1 sentinel (default 0.78, matches sim)")
    args = ap.parse_args()
    root = Path(args.root)

    parquets = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquets under {root}/data/chunk-*/")
    print(f"Found {len(parquets)} parquet files under {root}")

    changed = 0
    for i, p in enumerate(parquets):
        if augment_parquet(p, args.height):
            changed += 1
        if (i + 1) % 25 == 0 or i + 1 == len(parquets):
            print(f"  [{i + 1}/{len(parquets)}] processed (changed so far: {changed})")
    print(f"Augmented {changed}/{len(parquets)} parquets")

    update_info_json(root)
    print("Updated meta/info.json")


if __name__ == "__main__":
    main()
