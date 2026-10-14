"""Rewrite teleop.{left,right}_hand_joints in wigs parquets to 2-mode step signals.

Replaces the smooth physics-tracked joint state currently in those columns with
clean step transitions driven by the discrete commanded grip scalar
(action.psi0_18[6] for L, [13] for R, already in {0.0, 0.5, 1.0}). The new
encoding matches teleop_locomani_uni's convention so GR00T sees the same
action distribution across both datasets.

Mapping (right hand, scalar == 1.0 → grab; else → open):
    R_open = [0,  0,    0,    0, 0,    0, 0  ]
    R_grab = [0, -0.7, -0.7,  1, 1.5,  1, 1.5]
Left hand is sign-flipped (per teleop's _hand_to_grip convention). The
loco-manipulation task grasps with the right hand; the turn task with either,
so both hands are mapped.

Usage:
    .venv-openpi/bin/python scripts/data/rewrite_wigs_hand_joints.py \\
        --root data/lerobot/wigs/<NAME>_uni
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


R_OPEN = np.array([0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
R_GRAB = np.array([0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5], dtype=np.float32)
L_OPEN = -R_OPEN
L_GRAB = -R_GRAB
GRAB_THRESHOLD = 0.75


def rewrite(path: Path) -> tuple[int, int]:
    table = pq.read_table(path)
    act = np.array(table.column("action.psi0_18").to_pylist(), dtype=np.float32)
    l_cmd = act[:, 6]
    r_cmd = act[:, 13]

    l_hand = np.where((l_cmd >= GRAB_THRESHOLD)[:, None], L_GRAB[None, :], L_OPEN[None, :])
    r_hand = np.where((r_cmd >= GRAB_THRESHOLD)[:, None], R_GRAB[None, :], R_OPEN[None, :])

    l_arr = pa.array(l_hand.tolist(), type=pa.list_(pa.float32(), 7))
    r_arr = pa.array(r_hand.tolist(), type=pa.list_(pa.float32(), 7))

    table = table.set_column(table.schema.get_field_index("teleop.left_hand_joints"),
                             "teleop.left_hand_joints", l_arr)
    table = table.set_column(table.schema.get_field_index("teleop.right_hand_joints"),
                             "teleop.right_hand_joints", r_arr)

    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return int((l_cmd >= GRAB_THRESHOLD).sum()), int((r_cmd >= GRAB_THRESHOLD).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)

    parqs = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not parqs:
        raise SystemExit(f"No parquets under {root}/data/chunk-*/")
    print(f"Rewriting hand_joints in {len(parqs)} parquets under {root}")

    total_l = total_r = 0
    for i, p in enumerate(parqs):
        n_l, n_r = rewrite(p)
        total_l += n_l
        total_r += n_r
        if (i + 1) % 25 == 0 or i + 1 == len(parqs):
            print(f"  [{i + 1}/{len(parqs)}] cumulative grab-frames: left {total_l}, right {total_r}")
    print(f"Done. Total grab-frames: left {total_l}, right {total_r}")


if __name__ == "__main__":
    main()
