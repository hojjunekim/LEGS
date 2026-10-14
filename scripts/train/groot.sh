#!/usr/bin/env bash
# Fine-tune GR00T N1.6 on a LEGS task: projector and action head (the backbone stays frozen), EEF actions,
# chunk 16, global batch 24.
#
# Usage: TASK=locomani|turn bash scripts/train/groot.sh [--dry-run] [--set key=value ...]
#   PRESET          launcher preset (default: the paper preset for TASK)
#   GRAD_ACCUM      gradient accumulation steps (default: 1); the per-GPU batch is 24 / (NUM_GPUS * GRAD_ACCUM)
#   MASTER_PORT     torchrun port (default: 29501)
#   See common.sh for the shared variables.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_args "$@"
resolve_task
resolve_gpus
resolve_wandb

case "$TASK" in
    locomani) PRESET="${PRESET:-finetune_legs_locomani_uni_eef_gr00t_n16}" ;;
    turn) PRESET="${PRESET:-finetune_legs_turn_uni_eef_gr00t_n16}" ;;
esac
GLOBAL_BATCH=24
GRAD_ACCUM="${GRAD_ACCUM:-1}"
(( GRAD_ACCUM > 0 && GLOBAL_BATCH % (NUM_GPUS * GRAD_ACCUM) == 0 )) \
    || die "global batch $GLOBAL_BATCH is not a multiple of $NUM_GPUS GPUs x GRAD_ACCUM=$GRAD_ACCUM"
STEP_BATCH=$((GLOBAL_BATCH / GRAD_ACCUM))
PER_GPU_BATCH=$((STEP_BATCH / NUM_GPUS))
EXP="${EXP:-gr00t-n16-${DATASET##*/}-eef-$STEPS}"
RUN_DIR="$OUTPUT_DIR/gr00t-n16/$EXP"
MASTER_PORT="${MASTER_PORT:-29501}"
GR00T_PYTHON="${GR00T_PYTHON:-$PSI0_DIR/src/gr00t/.venv/bin/python}"
export GR00T_PYTHON
print_config TASK PRESET DATASET DATASET_DIR NUM_GPUS CUDA_VISIBLE_DEVICES GLOBAL_BATCH GRAD_ACCUM PER_GPU_BATCH \
    STEPS EXP RUN_DIR CKPT_DIR USE_WANDB

echo "== checks"
need -x "$GR00T_PYTHON" "run scripts/train/setup_envs.sh --groot"
need -f "$DATASET_DIR/meta/info.json" "run scripts/train/download.sh --dataset $DATASET"
need -f "$CKPT_DIR/$GROOT_CKPT/config.json" "run scripts/train/download.sh --groot"

# GR00T reads meta/modality.json; the paper runs swap in the EEF layout (the original is kept as
# modality_sonic.json). GR00T computes meta/relative_stats.json itself on the first run.
META="$DATASET_DIR/meta"
if [ -f "$META/info.json" ]; then
    [ -f "$META/modality_g1_eef_uni.json" ] || die "$META/modality_g1_eef_uni.json is missing"
    if ! cmp -s "$META/modality_g1_eef_uni.json" "$META/modality.json"; then
        echo "  modality: meta/modality.json <- meta/modality_g1_eef_uni.json"
        if [ "$DRY_RUN" != 1 ]; then
            [ -f "$META/modality_sonic.json" ] || cp "$META/modality.json" "$META/modality_sonic.json"
            cp "$META/modality_g1_eef_uni.json" "$META/modality.json"
        fi
    fi
fi
# The G1 EEF modality config reads meta/modality.json from DATASET_PATH in every rank.
export DATASET_PATH="$DATASET_DIR"
[ "$DRY_RUN" = 1 ] || ensure_psi0_dotenv

if [ "$USE_WANDB" = 1 ]; then
    WANDB_FLAG=use_wandb=true
else
    WANDB_FLAG=use_wandb=false
fi
LAUNCHER_PYTHON="$GR00T_PYTHON"
[ -x "$LAUNCHER_PYTHON" ] || LAUNCHER_PYTHON=python3

CMD=(
    "$LAUNCHER_PYTHON" baselines/gr00t-n1.6/finetune_gr00t.py
    --preset "$PRESET"
    --dataset-path "$DATASET_DIR"
    --base-model-path "$CKPT_DIR/$GROOT_CKPT"
    --output-dir "$RUN_DIR"
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES"
    --num-gpus "$NUM_GPUS"
    --master-port "$MASTER_PORT"
    --set max_steps="$STEPS"
    --set global_batch_size="$STEP_BATCH"
    --set gradient_accumulation_steps="$GRAD_ACCUM"
    --set "$WANDB_FLAG"
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
)
run_in "$PSI0_DIR" "${CMD[@]}"
# A dry run also shows the torchrun command the launcher resolves from the preset.
echo "== launcher dry run"
DRY_RUN_HINT="it needs PyYAML"
[ -x "$GR00T_PYTHON" ] || DRY_RUN_HINT="without the GR00T env it needs PyYAML in python3; run scripts/train/setup_envs.sh --groot"
(cd "$PSI0_DIR" && "${CMD[@]}" --dry-run) || echo "  (launcher dry run failed; $DRY_RUN_HINT)"
