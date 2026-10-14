"""Augment a universal LeRobot dataset with framework-agnostic 23-D upper-body
joint-space columns shared by psi0 / pi05 / gr00t-n1.6 fine-tunes.

Appends two packed columns to every parquet under
``<root>/data/chunk-*/episode_*.parquet``:

    observation.state_uni_20 (float32, [20])
    action.uni_23            (float32, [23])

Schema (joint-space upper body + grippers + height + base velocity):

    state_20  = left_arm(7) + right_arm(7) + rpy(3)
              + left_grip(1) + right_grip(1) + height(1)

    action_23 = state_20 layout(20, joints/rpy sourced from observation.state[t+1])
              + torso_vx(1) + torso_vy(1) + torso_vyaw(1)

Arms / waist are taken from OBSERVED joint readings (next-step) rather than
``action.wbc`` so wigs and teleop have matching action distributions:
``sonic_frame_builder.build_sonic_frame`` writes ``action.wbc = observation.state.copy()``
for wigs (no live WBC during sim recording), while teleop's ``action.wbc`` is
the real WBC command — which over-shoots especially on waist (σ commanded
≈0.22 vs σ observed ≈0.03). Cloning observed-next-state gives a
robot-realizable target and unifies the two sources.

Grip and base-command channels keep their existing action sources — both are
already discrete / clean step signals in ``action.psi0_18``:

    observation.state[15:22]   → left_arm  state
    observation.state[29:36]   → right_arm state
    observation.state[12:15]   → waist rpy state (reorder [13,14,12])
    observation.state_psi0[6]  → left_grip  state
    observation.state_psi0[13] → right_grip state
    observation.state_psi0[14] → height     state

    observation.state[15:22][t+1]   → left_arm  TARGET
    observation.state[29:36][t+1]   → right_arm TARGET
    observation.state[12:15][t+1]   → waist rpy TARGET (same reorder)
    action.psi0_18[6]               → left_grip  TARGET
    action.psi0_18[13]              → right_grip TARGET
    action.psi0_18[17]              → height     TARGET
    action.psi0_18[14:17]           → torso vx / vy / vyaw TARGET

Idempotent: skips parquets that already have both columns.

Usage:
    .venv-openpi/bin/python scripts/data/add_uni_23act_columns.py --root <root>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE_DIM = 20
ACTION_DIM = 23

LEFT_ARM = slice(15, 22)
RIGHT_ARM = slice(29, 36)
# waist is [yaw, roll, pitch] in dataset; we expose [roll, pitch, yaw]
RPY_REORDER = [13, 14, 12]

# Snap threshold for two-mode {0.5, 1.0} grip schema. teleop's _hand_to_grip
# already snaps at source; this also collapses wigs's stretch (0.0) → 0.5
# so both datasets share one grip distribution in uni_20/uni_23.
GRIP_SNAP_THRESHOLD = 0.75


def _snap_grip(g: np.ndarray) -> np.ndarray:
    return np.where(g >= GRIP_SNAP_THRESHOLD, 1.0, 0.5).astype(np.float32)


def _shift_next(arr: np.ndarray) -> np.ndarray:
    """Return arr[t+1]; last row duplicates arr[-1] (no extrapolation)."""
    out = np.empty_like(arr)
    out[:-1] = arr[1:]
    out[-1] = arr[-1]
    return out


def _pack_state(
    state_43: np.ndarray,
    l_grip: np.ndarray,
    r_grip: np.ndarray,
    height: np.ndarray,
) -> np.ndarray:
    n = state_43.shape[0]
    out = np.empty((n, STATE_DIM), dtype=np.float32)
    out[:, 0:7]   = state_43[:, LEFT_ARM]
    out[:, 7:14]  = state_43[:, RIGHT_ARM]
    out[:, 14:17] = state_43[:, RPY_REORDER]
    out[:, 17:18] = l_grip[:, None]
    out[:, 18:19] = r_grip[:, None]
    out[:, 19:20] = height[:, None]
    return out


def _pack_action(
    state_43_next: np.ndarray,
    l_grip_cmd: np.ndarray,
    r_grip_cmd: np.ndarray,
    height_cmd: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vyaw: np.ndarray,
) -> np.ndarray:
    n = state_43_next.shape[0]
    out = np.empty((n, ACTION_DIM), dtype=np.float32)
    out[:, 0:7]   = state_43_next[:, LEFT_ARM]
    out[:, 7:14]  = state_43_next[:, RIGHT_ARM]
    out[:, 14:17] = state_43_next[:, RPY_REORDER]
    out[:, 17:18] = l_grip_cmd[:, None]
    out[:, 18:19] = r_grip_cmd[:, None]
    out[:, 19:20] = height_cmd[:, None]
    out[:, 20:21] = vx[:, None]
    out[:, 21:22] = vy[:, None]
    out[:, 22:23] = vyaw[:, None]
    return out


def augment_parquet(path: Path) -> bool:
    table = pq.read_table(path)
    cols = set(table.column_names)
    if {"observation.state_uni_20", "action.uni_23"} <= cols:
        return False

    missing = {"observation.state", "observation.state_psi0", "action.psi0_18"} - cols
    if missing:
        raise SystemExit(f"{path}: missing required columns {missing}")

    state_43 = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    state_psi0 = np.asarray(table.column("observation.state_psi0").to_pylist(), dtype=np.float32)
    psi0_18 = np.asarray(table.column("action.psi0_18").to_pylist(), dtype=np.float32)

    n = state_43.shape[0]
    assert state_43.shape == (n, 43), state_43.shape
    assert state_psi0.shape == (n, 15), state_psi0.shape
    assert psi0_18.shape == (n, 18), psi0_18.shape

    state_43_next = _shift_next(state_43)

    state_20 = _pack_state(
        state_43=state_43,
        l_grip=_snap_grip(state_psi0[:, 6]),
        r_grip=_snap_grip(state_psi0[:, 13]),
        height=state_psi0[:, 14],
    )
    action_23 = _pack_action(
        state_43_next=state_43_next,
        l_grip_cmd=_snap_grip(psi0_18[:, 6]),
        r_grip_cmd=_snap_grip(psi0_18[:, 13]),
        height_cmd=psi0_18[:, 17],
        vx=psi0_18[:, 14],
        vy=psi0_18[:, 15],
        vyaw=psi0_18[:, 16],
    )

    table = table.append_column(
        "observation.state_uni_20",
        pa.array(state_20.tolist(), type=pa.list_(pa.float32(), STATE_DIM)),
    )
    table = table.append_column(
        "action.uni_23",
        pa.array(action_23.tolist(), type=pa.list_(pa.float32(), ACTION_DIM)),
    )

    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return True


# --- meta files -------------------------------------------------------------


STATE_NAMES_20 = (
    [f"left_arm_{i}" for i in range(7)]
    + [f"right_arm_{i}" for i in range(7)]
    + ["rpy_r", "rpy_p", "rpy_y", "left_grip", "right_grip", "height"]
)
ACTION_NAMES_23 = STATE_NAMES_20 + ["torso_vx", "torso_vy", "torso_vyaw"]


def update_info_json(root: Path) -> None:
    p = root / "meta/info.json"
    with open(p) as f:
        info = json.load(f)
    feats = info["features"]
    feats["observation.state_uni_20"] = {
        "dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES_20,
    }
    feats["action.uni_23"] = {
        "dtype": "float32", "shape": [ACTION_DIM], "names": ACTION_NAMES_23,
    }
    with open(p, "w") as f:
        json.dump(info, f, indent=4)


def write_modality_uni_23act(root: Path) -> None:
    """Sidecar consumed by gr00t-n1.6 / SONIC-style loaders that read
    meta/modality.json. psi0 / pi05 ignore it and take the dims and column keys
    as command-line overrides instead (``--model.odim=20 --model.action-dim=23``).
    """
    state_layout = [
        ("left_arm",   0,  7),
        ("right_arm",  7, 14),
        ("rpy",       14, 17),
        ("left_grip", 17, 18),
        ("right_grip",18, 19),
        ("height",    19, 20),
    ]
    action_layout = state_layout + [
        ("torso_vx",   20, 21),
        ("torso_vy",   21, 22),
        ("torso_vyaw", 22, 23),
    ]
    cfg = {
        "state": {
            k: {"start": s, "end": e, "original_key": "observation.state_uni_20"}
            for k, s, e in state_layout
        },
        "action": {
            k: {"start": s, "end": e, "original_key": "action.uni_23"}
            for k, s, e in action_layout
        },
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    with open(root / "meta/modality_uni_23act.json", "w") as f:
        json.dump(cfg, f, indent=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Universal LeRobot dataset root (modified in place)")
    args = ap.parse_args()
    root = Path(args.root)

    parquets = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquets under {root}/data/chunk-*/")
    print(f"Found {len(parquets)} parquet files under {root}")

    changed = 0
    for i, p in enumerate(parquets):
        if augment_parquet(p):
            changed += 1
        if (i + 1) % 25 == 0 or i + 1 == len(parquets):
            print(f"  [{i + 1}/{len(parquets)}] processed (changed so far: {changed})")
    print(f"Augmented {changed}/{len(parquets)} parquets")

    update_info_json(root)
    print("Updated meta/info.json")

    write_modality_uni_23act(root)
    print("Wrote meta/modality_uni_23act.json")


if __name__ == "__main__":
    main()
