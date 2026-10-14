#!/usr/bin/env bash
# Download the LEGS training datasets and the base weights of the three policies.
#
# Usage: bash scripts/train/download.sh [--datasets] [--psi0] [--pi05] [--groot] [--task locomani|turn]
#                                       [--dataset ID]... [--dry-run]
#   With no piece selected, everything is fetched. --task limits the datasets to one task.
#   --dataset fetches a dataset by its Hugging Face id (e.g. hojjunekim/sam_turn_uni, any set in the
#   README's dataset table) instead of the default ones; repeat it for several. Train on it with DATASET=ID.
#
# Layout under LEGS_CACHE (default ~/.cache/legs):
#   lerobot/<owner>/<dataset>                datasets (or $HF_LEROBOT_HOME/<owner>/<dataset> when it is set)
#   checkpoints/psi0/{pre.fast...,postpre...} psi0 VLM and action-head checkpoints
#   checkpoints/openpi/pi05_droid             pi0.5 DROID checkpoint (PyTorch)
#   checkpoints/GR00T-N1.6-3B                 GR00T N1.6 base model

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

GET_DATASETS=0 GET_PSI0=0 GET_PI05=0 GET_GROOT=0
TASKS=(locomani turn) DATASETS=() TASK_GIVEN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --datasets) GET_DATASETS=1 ;;
        --psi0) GET_PSI0=1 ;;
        --pi05) GET_PI05=1 ;;
        --groot) GET_GROOT=1 ;;
        --task) [ $# -ge 2 ] || die "--task needs locomani or turn"; TASKS=("$2") TASK_GIVEN=1; shift ;;
        --task=*) TASKS=("${1#--task=}") TASK_GIVEN=1 ;;
        --dataset) [ $# -ge 2 ] || die "--dataset needs a Hugging Face dataset id"; DATASETS+=("$2"); shift ;;
        --dataset=*) DATASETS+=("${1#--dataset=}") ;;
        --dry-run|--print-only) DRY_RUN=1 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done
if [ "${#DATASETS[@]}" -gt 0 ]; then
    [ "$TASK_GIVEN" = 0 ] || die "--task and --dataset do not combine; name each dataset with --dataset"
    for repo in "${DATASETS[@]}"; do
        [[ "$repo" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ && "$repo" != *..* ]] \
            || die "--dataset takes a full Hugging Face id such as hojjunekim/sam_turn_uni (got '$repo')"
    done
    GET_DATASETS=1
else
    for task in "${TASKS[@]}"; do
        case "$task" in
            locomani) DATASETS+=("$DATASET_LOCOMANI") ;;
            turn) DATASETS+=("$DATASET_TURN") ;;
            *) die "--task must be locomani or turn (got '$task')" ;;
        esac
    done
fi
if [ $((GET_DATASETS + GET_PSI0 + GET_PI05 + GET_GROOT)) -eq 0 ]; then
    GET_DATASETS=1 GET_PSI0=1 GET_PI05=1 GET_GROOT=1
fi

if command -v hf >/dev/null 2>&1; then
    HF=(hf)
elif [ -x "$PSI0_DIR/.venv-openpi/bin/hf" ]; then
    HF=("$PSI0_DIR/.venv-openpi/bin/hf")
elif command -v uvx >/dev/null 2>&1; then
    HF=(uvx --from huggingface_hub hf)
else
    die "no Hugging Face CLI found: pip install -U huggingface_hub"
fi

fetch() {
    echo "+ ${HF[*]} download $*"
    [ "$DRY_RUN" = 1 ] && return
    "${HF[@]}" download "$@" || die "download failed. Hugging Face limits the requests one address makes without an" \
        "account in 5 minutes; if that is the cause, log in with 'hf auth login' (a free account has a higher limit) or" \
        "wait 5 minutes and re-run, which skips the files already downloaded"
}

echo "== cache: $LEGS_CACHE (datasets: $HF_LEROBOT_HOME)"
if [ "$GET_DATASETS" = 1 ]; then
    # A dataset is hundreds of small files. Through Xet each file also costs an API request, which runs out a
    # logged-out download after about 500 files; without Xet a file costs only download requests, which are allowed
    # more often. HF_HUB_DISABLE_XET=0 turns Xet back on.
    for repo in "${DATASETS[@]}"; do
        HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
            fetch "$repo" --repo-type dataset --local-dir "$HF_LEROBOT_HOME/$repo"
    done
fi
if [ "$GET_PSI0" = 1 ]; then
    fetch USC-PSI-Lab/psi-model --include "$PSI0_VLM_CKPT/*" "$PSI0_HEAD_CKPT/*" --local-dir "$CKPT_DIR"
    # psi0 also reads the Qwen3-VL config and processor from the Hugging Face cache.
    fetch Qwen/Qwen3-VL-2B-Instruct --include "*.json" "*.txt" "*.jinja"
fi
if [ "$GET_PI05" = 1 ]; then
    fetch USC-PSI-Lab/psi-model --include "$PI05_CKPT/*" --local-dir "$CKPT_DIR"
fi
if [ "$GET_GROOT" = 1 ]; then
    fetch nvidia/GR00T-N1.6-3B --local-dir "$CKPT_DIR/$GROOT_CKPT"
fi
