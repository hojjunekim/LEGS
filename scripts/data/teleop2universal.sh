#!/usr/bin/env bash
# Existing GR00T-WBC teleop LeRobot dataset -> universal (Psi0 cols added in place).
#
# Usage: teleop2universal.sh <root> [--height 0.78]
#   <root> : LeRobot dataset root (modified in place: parquets gain
#            observation.state_psi0, action.psi0_18, observation.state_psi0_18;
#            meta/info.json + meta/stats.json regenerated)
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <root> [extra args forwarded to add_psi0_columns_teleop.py]" >&2
    exit 1
fi

ROOT=$1
shift
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# ROS 2 in the legs conda env sets PYTHONPATH to that env's site-packages, and conda deactivate leaves it set.
# The export env would then import the legs env's packages instead of its own.
unset PYTHONPATH
PY="${LEGS_DATA_PY:-$PROJECT/submodules/Psi0/.venv-openpi/bin/python}"
[[ -x "$PY" ]] || { echo "ERROR: $PY not found. Create the dataset-export env (SETUP.md) or set LEGS_DATA_PY." >&2; exit 1; }

"$PY" "$PROJECT/scripts/data/add_psi0_columns_teleop.py" --root "$ROOT" "$@"

"$PY" "$PROJECT/scripts/data/add_18state_column.py" --root "$ROOT"

# G1_EEF_UNI sidecar (meta/modality_g1_eef_uni.json) — slices the just-written
# observation.state_psi0_18 / action.psi0_18 into the 8-key 18-D EEF layout
# consumed by the gr00t-n1.6 finetune_*_eef_gr00t_n16 presets.
"$PY" "$PROJECT/scripts/data/add_g1_eef_uni_sidecar.py" --root "$ROOT"

"$PY" "$PROJECT/scripts/data/add_g1_loco_columns.py" --root "$ROOT"

# Universal 23-D upper-body-joint columns (observation.state_uni_20,
# action.uni_23) — shared input/output schema for psi0 / pi05 / gr00t-n1.6.
# See script docstring for the commanded-vs-observed rationale (wigs+teleop
# action distribution parity).
"$PY" "$PROJECT/scripts/data/add_uni_23act_columns.py" --root "$ROOT"

"$PY" "$PROJECT/submodules/Psi0/scripts/data/calc_modality_stats.py" --task-dir "$ROOT"

# norm_stats.json for pi05 quantile-norm — mirrors wigs2universal.sh so both
# datasets carry the same per-framework stats artifacts.
"$PY" "$PROJECT/scripts/data/calc_norm_stats.py" --task-dir "$ROOT"

echo "teleop2universal: done -> $ROOT"
