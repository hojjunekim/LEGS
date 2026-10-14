"""Augment a universal LeRobot dataset with GR00T-N1.6 `g1_loco_downstream` columns.

Appends two packed columns to every parquet under
``<root>/data/chunk-*/episode_*.parquet``:

    observation.state_g1_loco_32 (float32, [32])
    action.g1_loco_36            (float32, [36])

And writes ``meta/modality_g1_loco.json`` so the dataset can train the
``G1_LOCO_DOWNSTREAM`` embodiment of GR00T-N1.6 (modality config:
``submodules/Psi0/src/gr00t/gr00t/configs/modality/g1_locomanip.py``) alongside
the existing SONIC / Psi0 / Pi0.5 training targets.

Layout (matches g1_locomanip.py EXPECTED_*_KEYS; rpy is the G1 waist joints
reordered into roll-pitch-yaw, matching gr00t_serve_simple.py):

    state_32 = left_hand(7) + right_hand(7) + left_arm(7) + right_arm(7)
             + rpy(3)        + height(1)
    action_36 = state_32 layout(28)                                  ← arm+hand+rpy+height TARGETS from observation.state[t+1]
              + torso_vx(1) + torso_vy(1) + torso_vyaw(1) + target_yaw(1)

Sources (all already produced by build_sonic_lerobot.py / sonic teleop):
    observation.state[15:43]        → arm/hand state slots
    observation.state[12:15]        → rpy state (reorder [13,14,12])
    observation.state[15:43][t+1]   → arm/hand action TARGETS (next observed)
    observation.state[12:15][t+1]   → rpy action TARGETS (same reorder, next observed)
    teleop.planner_height           → height (sentinel -1 → --height fallback)
    teleop.planner_facing           → target_yaw = atan2(y, x)
    action.psi0_18[14:17]           → torso_vx/vy/vyaw (already computed there)

Action targets come from observed-next-state, NOT ``action.wbc``, so wigs and
teleop carry matching distributions. wigs's ``action.wbc`` is a copy of
``observation.state`` (no live WBC during sim recording), but teleop's
``action.wbc`` is the real WBC command — which over-shoots especially on
waist (σ commanded ≈0.22 vs σ observed ≈0.03). Cloning observed-next-state
gives a robot-realizable target and unifies the two sources. Matches the
rationale in ``add_uni_23act_columns.py``.

Idempotent: skips parquets that already have both columns.

Usage:
    .venv-openpi/bin/python scripts/data/add_g1_loco_columns.py --root <root> [--height 0.78]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE_DIM = 32
ACTION_DIM = 36

# Dataset slot indices into the 43-D SONIC joint vector
# (observation.state and action.wbc both follow SONIC_JOINT_ORDER).
WAIST = slice(12, 15)
LEFT_ARM = slice(15, 22)
LEFT_HAND = slice(22, 29)
RIGHT_ARM = slice(29, 36)
RIGHT_HAND = slice(36, 43)

# waist is [yaw, roll, pitch] in dataset; g1_locomanip's "rpy" expects
# [roll, pitch, yaw] (see gr00t_serve_simple.py:118 `[proprio[13:15], proprio[12:13]]`).
RPY_REORDER = [13, 14, 12]


def _pack_state(state_43: np.ndarray, height: np.ndarray) -> np.ndarray:
    n = state_43.shape[0]
    out = np.empty((n, STATE_DIM), dtype=np.float32)
    out[:, 0:7]   = state_43[:, LEFT_HAND]
    out[:, 7:14]  = state_43[:, RIGHT_HAND]
    out[:, 14:21] = state_43[:, LEFT_ARM]
    out[:, 21:28] = state_43[:, RIGHT_ARM]
    out[:, 28:31] = state_43[:, RPY_REORDER]
    out[:, 31:32] = height[:, None]
    return out


def _pack_action(
    state_43_next: np.ndarray,
    height: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vyaw: np.ndarray,
    target_yaw: np.ndarray,
) -> np.ndarray:
    n = state_43_next.shape[0]
    out = np.empty((n, ACTION_DIM), dtype=np.float32)
    out[:, 0:7]   = state_43_next[:, LEFT_HAND]
    out[:, 7:14]  = state_43_next[:, RIGHT_HAND]
    out[:, 14:21] = state_43_next[:, LEFT_ARM]
    out[:, 21:28] = state_43_next[:, RIGHT_ARM]
    out[:, 28:31] = state_43_next[:, RPY_REORDER]
    out[:, 31:32] = height[:, None]
    out[:, 32:33] = vx[:, None]
    out[:, 33:34] = vy[:, None]
    out[:, 34:35] = vyaw[:, None]
    out[:, 35:36] = target_yaw[:, None]
    return out


def _shift_next(arr: np.ndarray) -> np.ndarray:
    """Return arr[t+1]; last row duplicates arr[-1] (no extrapolation)."""
    out = np.empty_like(arr)
    out[:-1] = arr[1:]
    out[-1] = arr[-1]
    return out


def _height_from_planner(table: pa.Table, height_const: float) -> np.ndarray:
    n = table.num_rows
    cols = set(table.column_names)
    if "teleop.planner_height" in cols:
        hgt = np.asarray(table.column("teleop.planner_height").to_pylist(), dtype=np.float32).reshape(-1)
        return np.where(hgt > -0.5, hgt, height_const).astype(np.float32)
    return np.full(n, height_const, dtype=np.float32)


def _target_yaw_from_facing(table: pa.Table) -> np.ndarray:
    n = table.num_rows
    if "teleop.planner_facing" in set(table.column_names):
        f = np.asarray(table.column("teleop.planner_facing").to_pylist(), dtype=np.float32)
        return np.arctan2(f[:, 1], f[:, 0]).astype(np.float32)
    return np.zeros(n, dtype=np.float32)


def augment_parquet(path: Path, height_const: float) -> bool:
    table = pq.read_table(path)
    cols = set(table.column_names)
    if {"observation.state_g1_loco_32", "action.g1_loco_36"} <= cols:
        return False

    missing = {"observation.state", "action.psi0_18"} - cols
    if missing:
        raise SystemExit(f"{path}: missing required columns {missing}")

    state_43 = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    psi0_18  = np.asarray(table.column("action.psi0_18").to_pylist(),     dtype=np.float32)
    n = state_43.shape[0]
    assert state_43.shape == (n, 43), state_43.shape
    assert psi0_18.shape  == (n, 18), psi0_18.shape

    height     = _height_from_planner(table, height_const)
    target_yaw = _target_yaw_from_facing(table)
    vx, vy, vyaw = psi0_18[:, 14], psi0_18[:, 15], psi0_18[:, 16]

    state_43_next = _shift_next(state_43)

    state_32  = _pack_state(state_43, height)
    action_36 = _pack_action(state_43_next, height, vx, vy, vyaw, target_yaw)

    table = table.append_column(
        "observation.state_g1_loco_32",
        pa.array(state_32.tolist(), type=pa.list_(pa.float32(), STATE_DIM)),
    )
    table = table.append_column(
        "action.g1_loco_36",
        pa.array(action_36.tolist(), type=pa.list_(pa.float32(), ACTION_DIM)),
    )

    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return True


# --- meta files -------------------------------------------------------------


STATE_NAMES_32 = (
    [f"left_hand_{i}" for i in range(7)]
    + [f"right_hand_{i}" for i in range(7)]
    + [f"left_arm_{i}" for i in range(7)]
    + [f"right_arm_{i}" for i in range(7)]
    + ["rpy_r", "rpy_p", "rpy_y", "height"]
)
ACTION_NAMES_36 = STATE_NAMES_32 + ["torso_vx", "torso_vy", "torso_vyaw", "target_yaw"]


def update_info_json(root: Path) -> None:
    p = root / "meta/info.json"
    with open(p) as f:
        info = json.load(f)
    feats = info["features"]
    feats["observation.state_g1_loco_32"] = {
        "dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES_32,
    }
    feats["action.g1_loco_36"] = {
        "dtype": "float32", "shape": [ACTION_DIM], "names": ACTION_NAMES_36,
    }
    with open(p, "w") as f:
        json.dump(info, f, indent=4)


def write_modality_g1_loco(root: Path) -> None:
    """meta/modality_g1_loco.json: 6 state keys + 10 action keys matching
    g1_locomanip.py EXPECTED_*_KEYS. video key is rs_view (aliased to ego_view).
    """
    state_layout = [
        ("left_hand",  0,  7),
        ("right_hand", 7, 14),
        ("left_arm",  14, 21),
        ("right_arm", 21, 28),
        ("rpy",       28, 31),
        ("height",    31, 32),
    ]
    action_layout = state_layout + [
        ("torso_vx",   32, 33),
        ("torso_vy",   33, 34),
        ("torso_vyaw", 34, 35),
        ("target_yaw", 35, 36),
    ]
    cfg = {
        "state": {
            k: {"start": s, "end": e, "original_key": "observation.state_g1_loco_32"}
            for k, s, e in state_layout
        },
        "action": {
            k: {"start": s, "end": e, "original_key": "action.g1_loco_36"}
            for k, s, e in action_layout
        },
        "video": {"rs_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    with open(root / "meta/modality_g1_loco.json", "w") as f:
        json.dump(cfg, f, indent=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Universal LeRobot dataset root (modified in place)")
    ap.add_argument("--height", type=float, default=0.78,
                    help="Fallback when teleop.planner_height==-1 sentinel (default 0.78)")
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

    write_modality_g1_loco(root)
    print("Wrote meta/modality_g1_loco.json")


if __name__ == "__main__":
    main()
