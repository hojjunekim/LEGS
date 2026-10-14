#!/usr/bin/env bash
# Multi-source variant of wigs2universal.sh / sam2universal.sh.
# Combines N source dirs (each with its own ep range + task prompt) into one
# universal LeRobot v2.1 dataset, emitting >=2 task indices in meta/tasks.jsonl.
#
# Usage:
#   build_universal_multi.sh <wigs|sam> <save-root> \
#       --source-data-dir DIR --source-ep-from N --source-ep-to N --source-task STR \
#       [--source-data-dir DIR --source-ep-from N --source-ep-to N --source-task STR ...]
#
# Mode picks the renders-dir-name (wigs=final_renders, sam=baseline_renders);
# everything else is forwarded to build_sonic_lerobot.py.
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <wigs|sam> <save-root> --source-data-dir DIR --source-ep-from N --source-ep-to N --source-task STR [...]" >&2
    exit 1
fi

MODE=$1; shift
SAVE_ROOT=$1; shift

case "$MODE" in
    wigs) RENDERS_DIR=final_renders ;;
    sam)  RENDERS_DIR=baseline_renders ;;
    *) echo "mode must be 'wigs' or 'sam', got: $MODE" >&2; exit 1 ;;
esac

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# ROS 2 in the legs conda env sets PYTHONPATH to that env's site-packages, and conda deactivate leaves it set.
# The export env would then import the legs env's packages instead of its own.
unset PYTHONPATH
PY="${LEGS_DATA_PY:-$PROJECT/submodules/Psi0/.venv-openpi/bin/python}"
[[ -x "$PY" ]] || { echo "ERROR: $PY not found. Create the dataset-export env (SETUP.md) or set LEGS_DATA_PY." >&2; exit 1; }

mkdir -p "$(dirname "$SAVE_ROOT")"

"$PY" "$PROJECT/scripts/data/build_sonic_lerobot.py" \
    --save-root "$SAVE_ROOT" --overwrite \
    --renders-dir-name "$RENDERS_DIR" "$@"

"$PY" "$PROJECT/scripts/data/add_18state_column.py" --root "$SAVE_ROOT"
"$PY" "$PROJECT/scripts/data/add_g1_eef_uni_sidecar.py" --root "$SAVE_ROOT"
"$PY" "$PROJECT/scripts/data/add_g1_loco_columns.py" --root "$SAVE_ROOT"
"$PY" "$PROJECT/scripts/data/rewrite_wigs_hand_joints.py" --root "$SAVE_ROOT"
"$PY" "$PROJECT/scripts/data/add_uni_23act_columns.py" --root "$SAVE_ROOT"
"$PY" "$PROJECT/submodules/Psi0/scripts/data/calc_modality_stats.py" --task-dir "$SAVE_ROOT"
"$PY" "$PROJECT/scripts/data/calc_norm_stats.py" --task-dir "$SAVE_ROOT"

echo "build_universal_multi ($MODE): done -> $SAVE_ROOT"
