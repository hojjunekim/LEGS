#!/usr/bin/env bash
# Fine-tune pi0.5 on a LEGS task: full fine-tune from the DROID checkpoint, chunk 16, global batch 128.
#
# Usage: TASK=locomani|turn bash scripts/train/pi05.sh [--dry-run] [extra openpi flags...]
#   CONFIG          openpi train config (default: <dataset>_pi05, the config of DATASET)
#   SAVE_INTERVAL   checkpoint interval in steps (default: 2500)
#   See common.sh for the shared variables.
#
# The openpi trainer splits the global batch across GPUs and has no gradient accumulation, so the
# per-GPU batch is 128 / NUM_GPUS.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_args "$@"
resolve_task
resolve_gpus
resolve_wandb

# Each paper dataset has its own config. Serve the run with the same config: the server reads the
# normalization stats from the config's dataset.
CONFIG="${CONFIG:-${DATASET##*/}_pi05}"
GLOBAL_BATCH=128
(( GLOBAL_BATCH % NUM_GPUS == 0 )) || die "global batch $GLOBAL_BATCH does not split evenly over $NUM_GPUS GPUs"
PER_GPU_BATCH=$((GLOBAL_BATCH / NUM_GPUS))
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
EXP="${EXP:-pi05-${DATASET##*/}-full-$STEPS}"
RUN_DIR="$OUTPUT_DIR/openpi-05/$CONFIG/$EXP"
print_config TASK CONFIG DATASET DATASET_DIR NUM_GPUS CUDA_VISIBLE_DEVICES GLOBAL_BATCH PER_GPU_BATCH \
    STEPS SAVE_INTERVAL EXP RUN_DIR CKPT_DIR USE_WANDB

echo "== checks"
need -x "$PSI0_DIR/.venv-openpi/bin/python" "run scripts/train/setup_envs.sh --pi05"
need -f "$DATASET_DIR/meta/info.json" "run scripts/train/download.sh --dataset $DATASET"
need -f "$CKPT_DIR/$PI05_CKPT/model.safetensors" "run scripts/train/download.sh --pi05"

# openpi loads quantile norm stats from <dataset>/norm_stats.json. If it is missing, build it from the
# 18-D entries of meta/stats.json, which carry the same q01/q99 as the published file.
if [ ! -f "$DATASET_DIR/norm_stats.json" ]; then
    echo "  norm stats: $DATASET_DIR/norm_stats.json missing; writing it from meta/stats.json"
    if [ "$DRY_RUN" != 1 ]; then
        python3 - "$DATASET_DIR" <<'EOF'
import json
import sys

root = sys.argv[1]
stats = json.load(open(f"{root}/meta/stats.json"))
keys = {"state": "observation.state_psi0_18", "actions": "action.psi0_18"}
missing = [src for src in keys.values() if src not in stats]
if missing:
    sys.exit(f"meta/stats.json has no entry for {missing}; compute it with Psi0's scripts/data/calc_modality_stats.py")
norm = {name: {k: stats[src][k] for k in ("mean", "std", "q01", "q99")} for name, src in keys.items()}
with open(f"{root}/norm_stats.json", "w") as f:
    json.dump({"norm_stats": norm}, f, indent=2)
EOF
    fi
fi
[ "$DRY_RUN" = 1 ] || ensure_psi0_dotenv

ARGS=(
    --exp_name="$EXP"
    --data.repo-id="$DATASET_DIR"
    --pytorch_weight_path="$CKPT_DIR/$PI05_CKPT"
    --checkpoint_base_dir="$OUTPUT_DIR/openpi-05"
    --num_train_steps="$STEPS"
    --lr_schedule.decay_steps="$STEPS"
    --num_workers=8
    --resume
)
if [ "$USE_WANDB" = 1 ]; then
    ARGS+=(--wandb-enabled --project-name="$WANDB_PROJECT")
else
    ARGS+=(--no-wandb-enabled)
fi

export SAVE_INTERVAL
run_in "$PSI0_DIR" bash baselines/pi05/train_pi05.sh "$CONFIG" \
    "${ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
