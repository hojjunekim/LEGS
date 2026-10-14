# Shared settings for the LEGS training scripts. Sourced by the other scripts in this directory.
#
# Environment (all optional):
#   LEGS_CACHE       download cache for datasets and base weights (default: ~/.cache/legs)
#   HF_LEROBOT_HOME  LeRobot dataset root (default: $LEGS_CACHE/lerobot)
#   TASK             locomani | turn (default: locomani)
#   DATASET          Hugging Face dataset id (default: the paper dataset for TASK)
#   DATASET_DIR      local dataset directory (default: $HF_LEROBOT_HOME/$DATASET)
#   NUM_GPUS         GPUs to train on (default: every visible GPU)
#   STEPS            training steps (default: 20000)
#   OUTPUT_DIR       root of the run directories (default: submodules/Psi0/.runs)
#   EXP              experiment name (default: <model>-<dataset>-<scope>-<steps>)
#   WANDB_PROJECT    enables Weights & Biases logging under this project (uses your own login)
#   DRY_RUN=1        print the resolved config and command without running (same as --dry-run)

set -euo pipefail

# ROS 2 in the legs conda env sets PYTHONPATH to that env's site-packages, and conda deactivate leaves it set.
# The training envs would then import the legs env's torch instead of their own.
unset PYTHONPATH

LEGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PSI0_DIR="${PSI0_DIR:-$LEGS_ROOT/submodules/Psi0}"
LEGS_CACHE="${LEGS_CACHE:-$HOME/.cache/legs}"
CKPT_DIR="$LEGS_CACHE/checkpoints"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$LEGS_CACHE/lerobot}"
export PSI_HOME="${PSI_HOME:-$PSI0_DIR}"

# Paper datasets (200 episodes each, LeRobot v2.1).
DATASET_LOCOMANI="hojjunekim/legs_locomani_uni"
DATASET_TURN="hojjunekim/legs_turn_uni"

# Base weights.
PSI0_VLM_CKPT="psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
PSI0_HEAD_CKPT="psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"
PI05_CKPT="openpi/pi05_droid"
GROOT_CKPT="GR00T-N1.6-3B"

DRY_RUN="${DRY_RUN:-${PRINT_ONLY:-0}}"
EXTRA_ARGS=()

die() {
    echo "error: $*" >&2
    exit 1
}

# Splits the command line into --dry-run and the flags forwarded to the trainer.
parse_args() {
    local arg
    for arg in "$@"; do
        case "$arg" in
            --dry-run|--print-only) DRY_RUN=1 ;;
            *) EXTRA_ARGS+=("$arg") ;;
        esac
    done
}

resolve_task() {
    TASK="${TASK:-locomani}"
    case "$TASK" in
        locomani) DATASET="${DATASET:-$DATASET_LOCOMANI}" ;;
        turn) DATASET="${DATASET:-$DATASET_TURN}" ;;
        *) die "TASK must be locomani or turn (got '$TASK')" ;;
    esac
    DATASET_DIR="${DATASET_DIR:-$HF_LEROBOT_HOME/$DATASET}"
    STEPS="${STEPS:-20000}"
    OUTPUT_DIR="${OUTPUT_DIR:-$PSI0_DIR/.runs}"
}

# Sets NUM_GPUS and exports CUDA_VISIBLE_DEVICES to match it.
resolve_gpus() {
    local visible="" i
    local -a ids=()
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        visible="${CUDA_VISIBLE_DEVICES// /}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        visible="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
    fi
    [ -n "$visible" ] && IFS=',' read -r -a ids <<< "$visible"
    if [ -n "${NUM_GPUS:-}" ]; then
        if [ "${#ids[@]}" -lt "$NUM_GPUS" ]; then
            [ "$DRY_RUN" = 1 ] || die "NUM_GPUS=$NUM_GPUS but only ${#ids[@]} GPU(s) are visible"
            ids=()
            for ((i = 0; i < NUM_GPUS; i++)); do ids+=("$i"); done
        fi
        ids=("${ids[@]:0:$NUM_GPUS}")
    fi
    NUM_GPUS="${#ids[@]}"
    [ "$NUM_GPUS" -gt 0 ] || die "no GPU found; set CUDA_VISIBLE_DEVICES or NUM_GPUS"
    CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${ids[*]}")"
    export CUDA_VISIBLE_DEVICES
}

# Weights & Biases stays off unless WANDB_PROJECT is set.
resolve_wandb() {
    if [ -n "${WANDB_PROJECT:-}" ]; then
        USE_WANDB=1
    else
        USE_WANDB=0
        export WANDB_MODE=disabled
    fi
}

# Psi0's entry points refuse to start without a non-empty .env; exported variables take precedence over it.
ensure_psi0_dotenv() {
    [ -s "$PSI0_DIR/.env" ] && return 0
    printf '# Created by the LEGS training scripts. Variables exported in the shell take precedence.\nPSI_HOME=%s\n' \
        "$PSI0_DIR" > "$PSI0_DIR/.env"
}

# need -f|-d PATH HINT: fails when PATH is missing (a dry run only reports it).
need() {
    if ! test "$1" "$2"; then
        [ "$DRY_RUN" = 1 ] || die "missing $2 ($3)"
        echo "  [missing] $2 ($3)"
    fi
}

print_config() {
    local key
    echo "== resolved config"
    for key in "$@"; do
        printf '  %-20s %s\n' "$key" "${!key}"
    done
}

# run_in DIR CMD...: prints the command, then runs it in DIR unless this is a dry run.
run_in() {
    local dir="$1" line="" arg
    shift
    echo "== command (in $dir)"
    for arg in "$@"; do
        if [[ "$arg" == --* && -n "$line" ]]; then
            printf '  %s \\\n' "$line"
            line=""
        fi
        line+="${line:+ }$(printf '%q' "$arg")"
    done
    printf '  %s\n' "$line"
    if [ "$DRY_RUN" = 1 ]; then
        echo "== dry run: nothing started"
        return 0
    fi
    cd "$dir"
    exec "$@"
}
