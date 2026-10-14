"""Write meta/modality_g1_eef_uni.json sidecar for the gr00t-n1.6 G1_EEF_UNI
embodiment (18-state / 18-action psi0 EEF schema, 8-key split).

No new parquet columns — `observation.state_psi0_18` and `action.psi0_18`
must already exist (produced by scripts/data/add_18state_column.py for state
and by the original psi0 builder for action).

Schema split (mirrors action.psi0_18 layout in add_18state_column.py):
    [0:3]   left_eef_xyz
    [3:6]   left_eef_rpy
    [6:7]   left_grip
    [7:10]  right_eef_xyz
    [10:13] right_eef_rpy
    [13:14] right_grip
    [14:17] base_vel       (state: prev-step cmd, action: current cmd)
    [17:18] height

scripts/train/groot.sh swaps meta/modality.json -> modality_g1_eef_uni.json before
launching (gr00t loader hardcodes meta/modality.json), so this sidecar is
what the g1_eef_uni.py modality config in Psi0 will validate against.

Usage:
    python scripts/data/add_g1_eef_uni_sidecar.py \
        --root data/lerobot/wigs/<NAME>_uni
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_sidecar() -> dict:
    state_layout = [
        ("left_eef_xyz",  0,  3),
        ("left_eef_rpy",  3,  6),
        ("left_grip",     6,  7),
        ("right_eef_xyz", 7, 10),
        ("right_eef_rpy", 10, 13),
        ("right_grip",    13, 14),
        ("base_vel",      14, 17),
        ("height",        17, 18),
    ]
    action_layout = state_layout  # identical 18-D layout

    state = {
        k: {"start": s, "end": e, "original_key": "observation.state_psi0_18"}
        for k, s, e in state_layout
    }
    action = {
        k: {"start": s, "end": e, "original_key": "action.psi0_18"}
        for k, s, e in action_layout
    }
    return {
        "state": state,
        "action": action,
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="LeRobot dataset root")
    args = ap.parse_args()
    root = Path(args.root)
    meta_dir = root / "meta"
    if not meta_dir.is_dir():
        raise SystemExit(f"no meta/ under {root}")

    info_path = meta_dir / "info.json"
    info = json.loads(info_path.read_text())
    feats = info.get("features", {})
    for col in ("observation.state_psi0_18", "action.psi0_18"):
        if col not in feats:
            raise SystemExit(
                f"{info_path}: missing required column '{col}' in features"
            )
        shape = feats[col].get("shape")
        if shape != [18]:
            raise SystemExit(f"{col} shape {shape} != [18]")

    out = meta_dir / "modality_g1_eef_uni.json"
    out.write_text(json.dumps(build_sidecar(), indent=4))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
