#!/usr/bin/env bash
# Fine-tune psi0 on a LEGS task: action head only, 18-D EEF state and action, chunk 30, global batch 256.
#
# Usage: TASK=locomani|turn bash scripts/train/psi0.sh [--dry-run] [extra psi0 flags...]
#   PER_GPU_BATCH   per-GPU batch (default: min(64, 256 / NUM_GPUS)); gradient accumulation makes up the rest
#   See common.sh for the shared variables.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_args "$@"
resolve_task
resolve_gpus
resolve_wandb

GLOBAL_BATCH=256
PER_GPU_BATCH="${PER_GPU_BATCH:-$((GLOBAL_BATCH / NUM_GPUS < 64 ? GLOBAL_BATCH / NUM_GPUS : 64))}"
(( PER_GPU_BATCH > 0 && GLOBAL_BATCH % (PER_GPU_BATCH * NUM_GPUS) == 0 )) \
    || die "global batch $GLOBAL_BATCH is not a multiple of $PER_GPU_BATCH x $NUM_GPUS GPUs; use a GPU count that divides it or set PER_GPU_BATCH"
GRAD_ACCUM=$((GLOBAL_BATCH / (PER_GPU_BATCH * NUM_GPUS)))
EXP="${EXP:-psi0-${DATASET##*/}-head-$STEPS}"

# The base script reads <DATA_ROOT>/<repo id>.
if [ "$DATASET_DIR" = "$HF_LEROBOT_HOME/$DATASET" ]; then
    export DATA_ROOT="$HF_LEROBOT_HOME"
    REPO_ID="$DATASET"
else
    export DATA_ROOT="$(dirname "$DATASET_DIR")"
    REPO_ID="$(basename "$DATASET_DIR")"
fi
RUN_DIR="$OUTPUT_DIR/finetune"
print_config TASK DATASET DATASET_DIR NUM_GPUS CUDA_VISIBLE_DEVICES GLOBAL_BATCH PER_GPU_BATCH GRAD_ACCUM \
    STEPS EXP RUN_DIR CKPT_DIR USE_WANDB

echo "== checks"
need -x "$PSI0_DIR/.venv-psi/bin/python" "run scripts/train/setup_envs.sh --psi0"
need -f "$DATASET_DIR/meta/info.json" "run scripts/train/download.sh --dataset $DATASET"
need -d "$CKPT_DIR/$PSI0_VLM_CKPT" "run scripts/train/download.sh --psi0"
need -d "$CKPT_DIR/$PSI0_HEAD_CKPT" "run scripts/train/download.sh --psi0"

# Normalization uses the 18-D state and action entries of meta/stats.json; compute them if they are missing.
STATS="$DATASET_DIR/meta/stats.json"
if ! python3 -c 'import json, sys; s = json.load(open(sys.argv[1])); sys.exit(0 if all(k in s for k in sys.argv[2:]) else 1)' \
        "$STATS" observation.state_psi0_18 action.psi0_18 2>/dev/null; then
    echo "  norm stats: $STATS lacks the 18-D entries; computing them with scripts/data/calc_modality_stats.py"
    if [ "$DRY_RUN" != 1 ]; then
        [ -f "$STATS" ] && [ ! -f "$STATS.orig" ] && cp "$STATS" "$STATS.orig"
        (cd "$PSI0_DIR" && .venv-psi/bin/python scripts/data/calc_modality_stats.py --task-dir "$DATASET_DIR")
    fi
fi
[ "$DRY_RUN" = 1 ] || ensure_psi0_dotenv

ARGS=(
    --train.max_training_steps="$STEPS"
    --train.train_batch_size="$PER_GPU_BATCH"
    --train.gradient_accumulation_steps="$GRAD_ACCUM"
    --train.num_workers=8
    --train.output_dir="$OUTPUT_DIR"
    # Camera, state and action keys of the LEGS datasets, and their 18-D stats.
    --data.transform.repack.image-keys observation.images.ego_view
    --data.transform.repack.state-key=observation.state_psi0_18
    --data.transform.repack.action-key=action.psi0_18
    --data.transform.field.stat-state-key=observation.state_psi0_18
    --data.transform.field.stat-action-key=action.psi0_18
    --model.odim=18
    --model.model_name_or_path="$CKPT_DIR/$PSI0_VLM_CKPT"
    --model.pretrained-action-header-path="$CKPT_DIR/$PSI0_HEAD_CKPT"
)
# The base script turns W&B on when WANDB_PROJECT is set.
run_in "$PSI0_DIR" bash scripts/train/psi0/finetune-real-psi0-18act-eef.sh "$REPO_ID" "$EXP" \
    "${ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
