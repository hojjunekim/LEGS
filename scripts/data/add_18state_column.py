"""Augment an existing Psi0 LeRobot dataset with an 18-D state column.

Adds two new columns to every parquet under <root>/data/chunk-*/episode_*.parquet:
    observation.state_psi0_18 (float32, shape=[18])
    states_18                 (float32, shape=[18])  # alias for stat-key compat

Layout (matches action.psi0_18):
    [0:14]  = observation.state_psi0[0:14]            # L_eef_6d, L_grip, R_eef_6d, R_grip
    [14:17] = action.psi0_18[t-1][14:17]              # previous-step base cmd (vx, vy, vyaw)
    [17]    = observation.state_psi0[14]              # height
For t=0 (first row of each episode), base-cmd dims are filled with zeros.

Also:
    - Updates meta/info.json with the two new features.
    - Writes meta/modality_psi0_18state.json (parallel to modality_psi0.json).

Usage:
    .venv-openpi/bin/python scripts/data/add_18state_column.py \
        --root data/lerobot/wigs/<NAME>_uni
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def augment_parquet(path: Path) -> bool:
    table = pq.read_table(path)
    cols = table.column_names
    if "observation.state_psi0_18" in cols and "states_18" in cols:
        return False

    state = np.asarray(table["observation.state_psi0"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action.psi0_18"].to_pylist(), dtype=np.float32)
    n = state.shape[0]
    assert state.shape == (n, 15), state.shape
    assert action.shape == (n, 18), action.shape

    base_prev = np.zeros((n, 3), dtype=np.float32)
    if n > 1:
        base_prev[1:] = action[:-1, 14:17]

    state_18 = np.empty((n, 18), dtype=np.float32)
    state_18[:, 0:14] = state[:, 0:14]
    state_18[:, 14:17] = base_prev
    state_18[:, 17] = state[:, 14]

    arr = pa.array(state_18.tolist(), type=pa.list_(pa.float32(), 18))
    if "observation.state_psi0_18" not in cols:
        table = table.append_column("observation.state_psi0_18", arr)
    if "states_18" not in cols:
        table = table.append_column("states_18", arr)

    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return True


def update_info_json(root: Path) -> None:
    p = root / "meta/info.json"
    with open(p) as f:
        info = json.load(f)
    feats = info["features"]

    names_18 = [
        "L_eef_x", "L_eef_y", "L_eef_z", "L_eef_roll", "L_eef_pitch", "L_eef_yaw", "L_grip",
        "R_eef_x", "R_eef_y", "R_eef_z", "R_eef_roll", "R_eef_pitch", "R_eef_yaw", "R_grip",
        "vx", "vy", "vyaw", "height",
    ]
    entry = {"dtype": "float32", "shape": [18], "names": names_18}
    feats["observation.state_psi0_18"] = entry
    feats["states_18"] = entry

    with open(p, "w") as f:
        json.dump(info, f, indent=4)


def write_modality_18state(root: Path) -> None:
    out = root / "meta/modality_psi0_18state.json"
    cfg = {
        "state": {"all": {"start": 0, "end": 18, "original_key": "observation.state_psi0_18"}},
        "action": {"all": {"start": 0, "end": 18, "original_key": "action.psi0_18"}},
        "video": {"egocentric": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    with open(out, "w") as f:
        json.dump(cfg, f, indent=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LeRobot dataset root")
    args = ap.parse_args()
    root = Path(args.root)

    parquets = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    print(f"Found {len(parquets)} parquet files under {root}")

    changed = 0
    for i, p in enumerate(parquets):
        if augment_parquet(p):
            changed += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(parquets)}] processed (changed so far: {changed})")
    print(f"Augmented {changed}/{len(parquets)} parquets")

    update_info_json(root)
    print("Updated meta/info.json")

    write_modality_18state(root)
    print("Wrote meta/modality_psi0_18state.json")


if __name__ == "__main__":
    main()
