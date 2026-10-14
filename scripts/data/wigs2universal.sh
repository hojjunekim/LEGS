#!/usr/bin/env bash
# Sim recordings -> universal LeRobot v2.1 dataset (GR00T + Psi0 cols).
#
# Usage: wigs2universal.sh <data-dir> <save-root> [token-csv-dir] [task] [-- <build args>]
#   <data-dir>      : raw sim recordings (episode_XXXX.json + _scenario.json + final_renders/)
#                     produced by simulate_object.py + offline_renderer.py
#   <save-root>     : output LeRobot dataset root (will be overwritten)
#   [token-csv-dir] : optional dir scanned recursively for token_state.csv.
#                     If omitted or contains no CSVs, action.motion_token is
#                     zero-filled (OK for psi0/pi05/GR00T N1.6; breaks SONIC).
#                     Pass "" to skip while still supplying [task].
#   [task]          : optional override for per-episode language_instruction.
#                     Needed when the aug pipeline swaps the rendered scene
#                     (e.g. apple_to_box_*) but scenario.json carries the
#                     original orange/plate prompt.
#   -- <build args> : passed through to build_sonic_lerobot.py, e.g.
#                     -- --allow-skip --allow-missing-frames
set -euo pipefail

POS=() EXTRA=()
while [[ $# -gt 0 ]]; do
    if [[ $1 == -- ]]; then shift; EXTRA=("$@"); break; fi
    POS+=("$1"); shift
done
if [[ ${#POS[@]} -lt 2 || ${#POS[@]} -gt 4 ]]; then
    echo "Usage: $0 <data-dir> <save-root> [token-csv-dir] [task] [-- <build args>]" >&2
    exit 1
fi

DATA_DIR=${POS[0]}
SAVE_ROOT=${POS[1]}
TOKEN_CSV_DIR=${POS[2]:-}
TASK=${POS[3]:-}
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# ROS 2 in the legs conda env sets PYTHONPATH to that env's site-packages, and conda deactivate leaves it set.
# The export env would then import the legs env's packages instead of its own.
unset PYTHONPATH
PY="${LEGS_DATA_PY:-$PROJECT/submodules/Psi0/.venv-openpi/bin/python}"
[[ -x "$PY" ]] || { echo "ERROR: $PY not found. Create the dataset-export env (SETUP.md) or set LEGS_DATA_PY." >&2; exit 1; }

mkdir -p "$(dirname "$SAVE_ROOT")"

BUILD_ARGS=(--data-dir "$DATA_DIR" --save-root "$SAVE_ROOT" --overwrite)
if [[ -n "$TOKEN_CSV_DIR" ]]; then
    [[ -d "$TOKEN_CSV_DIR" ]] || { echo "ERROR: token-csv-dir $TOKEN_CSV_DIR does not exist (pass \"\" to zero-fill)" >&2; exit 1; }
    BUILD_ARGS+=(--token-csv-dir "$TOKEN_CSV_DIR")
fi
if [[ -n "$TASK" ]]; then
    BUILD_ARGS+=(--task "$TASK")
fi
"$PY" "$PROJECT/scripts/data/build_sonic_lerobot.py" "${BUILD_ARGS[@]}" "${EXTRA[@]}"

"$PY" "$PROJECT/scripts/data/add_18state_column.py" --root "$SAVE_ROOT"

# G1_EEF_UNI sidecar (meta/modality_g1_eef_uni.json) — slices the just-written
# observation.state_psi0_18 / action.psi0_18 into the 8-key 18-D EEF layout
# consumed by the gr00t-n1.6 finetune_*_eef_gr00t_n16 presets.
"$PY" "$PROJECT/scripts/data/add_g1_eef_uni_sidecar.py" --root "$SAVE_ROOT"

"$PY" "$PROJECT/scripts/data/add_g1_loco_columns.py" --root "$SAVE_ROOT"

# Replace the physics-tracked smooth hand joints with clean step signals
# driven by action.psi0_18[6/13] (matches teleop's discrete grip distribution).
# Must run before calc_modality_stats so the SONIC stats reflect the final
# distribution rather than the raw simulator output.
"$PY" "$PROJECT/scripts/data/rewrite_wigs_hand_joints.py" --root "$SAVE_ROOT"

# Universal 23-D upper-body-joint columns (observation.state_uni_20,
# action.uni_23) — shared input/output schema for psi0 / pi05 / gr00t-n1.6.
# Reads observed joints + observation.state_psi0 grippers + action.psi0_18
# base/grip TARGETS — see script docstring for the commanded-vs-observed
# rationale (wigs+teleop action distribution parity).
"$PY" "$PROJECT/scripts/data/add_uni_23act_columns.py" --root "$SAVE_ROOT"

"$PY" "$PROJECT/submodules/Psi0/scripts/data/calc_modality_stats.py" --task-dir "$SAVE_ROOT"

# norm_stats.json for pi05 quantile-norm (state=observation.state_psi0_18[18],
# actions=action.psi0_18[18]). meta/stats.json above covers psi0 bounds-norm.
"$PY" "$PROJECT/scripts/data/calc_norm_stats.py" --task-dir "$SAVE_ROOT"

echo "wigs2universal: done -> $SAVE_ROOT"
