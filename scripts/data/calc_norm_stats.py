#!/usr/bin/env python3
"""Generate norm_stats.json for openpi pi05 fine-tune.

Reads observation.state_psi0_18 (18-D) and action.psi0_18 (18-D) from all
parquets in <task-dir>/data/, computes mean/std/q01/q99 per dim, and writes
<task-dir>/norm_stats.json in openpi's quantile-norm format:

    {"norm_stats": {"state":   {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
                    "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}}}

The keys match LeRobotUniversalDataConfig's repack:
    states  <- observation.state_psi0_18
    actions <- action.psi0_18

18-D state matches scripts/train/psi0.sh (--data.transform.repack.state-key
= observation.state_psi0_18 + --model.odim=18) so pi05 and psi0 see the same
observation space — fair head-to-head training.

Required for pi05 full FT. Harmless for psi0 (uses meta/stats.json) and
GR00T/SONIC (ignores norm_stats.json).
"""
import argparse
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq


STATE_KEY = "observation.state_psi0_18"
ACTION_KEY = "action.psi0_18"


def stack_col(parquets, col):
    pieces = [np.asarray(pq.read_table(p, columns=[col]).column(col).to_pylist(),
                         dtype=np.float64) for p in parquets]
    return np.concatenate(pieces, axis=0)


def stats(arr):
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std":  arr.std(axis=0).tolist(),
        "q01":  np.quantile(arr, 0.01, axis=0).tolist(),
        "q99":  np.quantile(arr, 0.99, axis=0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    args = ap.parse_args()

    root = pathlib.Path(args.task_dir)
    parquets = sorted(root.glob("data/**/*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquets found under {root}/data")

    state = stack_col(parquets, STATE_KEY)
    action = stack_col(parquets, ACTION_KEY)

    out = {"norm_stats": {"state": stats(state), "actions": stats(action)}}
    (root / "norm_stats.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {root / 'norm_stats.json'}  "
          f"state={state.shape}  action={action.shape}  parquets={len(parquets)}")


if __name__ == "__main__":
    main()
