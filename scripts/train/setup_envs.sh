#!/usr/bin/env bash
# Build the training environments inside submodules/Psi0 with uv. Safe to re-run: finished steps are skipped
# and uv only installs what is missing.
#
# Usage: bash scripts/train/setup_envs.sh [--psi0] [--pi05] [--groot] [--dry-run]
#   With no env selected, all three are built.
#   --psi0   .venv-psi        psi0 (torch 2.7.0, flash-attn 2.7.4.post1)
#   --pi05   .venv-openpi     pi0.5 (openpi, torch 2.7.1, JAX 0.5.3); also the dataset-export env
#                             (lerobot < 0.3 and gear_sonic from submodules/GR00T-WholeBodyControl)
#   --groot  src/gr00t/.venv  GR00T N1.6 (torch 2.7.0, flash-attn 2.7.4.post1; its policy server's packages)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BUILD_PSI0=0 BUILD_PI05=0 BUILD_GROOT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --psi0) BUILD_PSI0=1 ;;
        --pi05) BUILD_PI05=1 ;;
        --groot) BUILD_GROOT=1 ;;
        --dry-run|--print-only) DRY_RUN=1 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done
if [ $((BUILD_PSI0 + BUILD_PI05 + BUILD_GROOT)) -eq 0 ]; then
    BUILD_PSI0=1 BUILD_PI05=1 BUILD_GROOT=1
fi

command -v uv >/dev/null 2>&1 || die "uv not found: https://docs.astral.sh/uv/getting-started/installation/"
[ -f "$PSI0_DIR/pyproject.toml" ] || die "submodules/Psi0 is empty: check out the submodules recursively (SETUP.md, section 1)"
# Psi0's uv workspace points at third_party/SIMPLE, so uv needs it on disk even though training does not use it.
[ -n "$(ls -A "$PSI0_DIR/third_party/SIMPLE" 2>/dev/null)" ] \
    || die "submodules/Psi0/third_party/SIMPLE is empty: check out the submodules recursively (SETUP.md, section 1)"
cd "$PSI0_DIR"
export GIT_LFS_SKIP_SMUDGE=1

run() {
    echo "+ $*"
    [ "$DRY_RUN" = 1 ] || "$@"
}

# has VENV MODULE: true when the module imports in that venv.
has() {
    [ -x "$1/bin/python" ] && PYTHONDONTWRITEBYTECODE=1 "$1/bin/python" -c "import $2" >/dev/null 2>&1
}

make_venv() {
    if [ -x "$1/bin/python" ]; then
        echo "  $1 exists"
    else
        run uv venv "$1" --python 3.10
    fi
}

FLASH_ATTN=2.7.4.post1

# flash_attn_wheel VENV: URL of the prebuilt flash-attn wheel for the venv's torch, CUDA, C++ ABI and Python.
flash_attn_wheel() {
    PYTHONDONTWRITEBYTECODE=1 "$1/bin/python" -c "import platform, sys, torch
t = torch.__version__.split('+')[0].split('.')
py = 'cp%d%d' % sys.version_info[:2]
tag = 'cu%storch%s.%scxx11abi%s' % (torch.version.cuda.split('.')[0], t[0], t[1],
                                   str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())
print('https://github.com/Dao-AILab/flash-attention/releases/download/v$FLASH_ATTN/flash_attn-$FLASH_ATTN+%s-%s-%s-linux_%s.whl'
      % (tag, py, py, platform.machine()))" 2>/dev/null
}

# install_flash_attn VENV: installs flash-attn from its prebuilt wheel, so no CUDA toolkit is needed (flash-attn's own
# installer stops at an nvcc older than CUDA 11.7 before it looks for that wheel). Without a matching wheel it builds
# from source, which needs nvcc from a CUDA 12 toolkit (CUDA_HOME) and takes a while.
install_flash_attn() {
    local wheel
    if has "$1" flash_attn; then
        echo "  flash_attn installed"
        return
    fi
    if [ ! -x "$1/bin/python" ]; then  # dry run before the venv exists
        echo "+ uv pip install <the flash-attn $FLASH_ATTN wheel that matches the venv's torch>"
        return
    fi
    wheel="$(flash_attn_wheel "$1")" || wheel=""
    if [ -n "$wheel" ] && curl -fsIL -o /dev/null "$wheel"; then
        VIRTUAL_ENV="$PSI0_DIR/$1" run uv pip install "$wheel"
    else
        echo "  no prebuilt flash-attn wheel matches this torch, CUDA and Python; building it from source"
        VIRTUAL_ENV="$PSI0_DIR/$1" run uv pip install "flash_attn==$FLASH_ATTN" --no-build-isolation
    fi
}

if [ "$BUILD_PSI0" = 1 ]; then
    echo "== psi0: .venv-psi"
    make_venv .venv-psi
    # --inexact keeps flash-attn, which is installed outside the lock file.
    VIRTUAL_ENV="$PSI0_DIR/.venv-psi" run uv sync --active --inexact --group serve --group viz --group psi \
        --index-strategy unsafe-best-match
    install_flash_attn .venv-psi
fi

if [ "$BUILD_PI05" = 1 ]; then
    echo "== pi0.5: .venv-openpi"
    make_venv .venv-openpi
    VIRTUAL_ENV="$PSI0_DIR/.venv-openpi" run uv pip install -e .
    VIRTUAL_ENV="$PSI0_DIR/.venv-openpi" run uv pip install -e src/openpi/openpi-client
    VIRTUAL_ENV="$PSI0_DIR/.venv-openpi" run uv pip install -r baselines/pi05/requirements-openpi.txt
    # openpi's PyTorch models need patched transformers files; re-applied whenever transformers is reinstalled.
    if PYTHONDONTWRITEBYTECODE=1 .venv-openpi/bin/python -c "from transformers.models.siglip import check; \
            import sys; sys.exit(0 if check.check_whether_transformers_replace_is_installed_correctly() else 1)" \
            >/dev/null 2>&1; then
        echo "  transformers patch applied"
    else
        run cp -r src/openpi/models_pytorch/transformers_replace/. .venv-openpi/lib/python3.10/site-packages/transformers/
    fi
    # Dataset export (scripts/data) runs in this env too.
    if [ -f ../GR00T-WholeBodyControl/gear_sonic/pyproject.toml ]; then
        VIRTUAL_ENV="$PSI0_DIR/.venv-openpi" run uv pip install -e ../GR00T-WholeBodyControl/gear_sonic --no-deps
    else
        echo "  submodules/GR00T-WholeBodyControl is empty; skipping gear_sonic (only the dataset export needs it)"
    fi
fi

if [ "$BUILD_GROOT" = 1 ]; then
    echo "== GR00T N1.6: src/gr00t/.venv"
    # The lock file has only flash-attn's source release, so the sync skips it and it is installed from its wheel.
    # --inexact keeps that wheel on later runs.
    (cd src/gr00t && run uv sync --inexact --no-install-package flash-attn)
    install_flash_attn src/gr00t/.venv
    # The policy server (baselines/gr00t-n1.6/deploy_gr00t_simple.sh) also needs packages that are not in GR00T's lock
    # file: Psi0's serve group and python-dotenv, at the versions in Psi0's uv.lock.
    if has src/gr00t/.venv "fastapi, uvicorn, dotenv"; then
        echo "  server packages installed"
    else
        VIRTUAL_ENV="$PSI0_DIR/src/gr00t/.venv" run uv pip install fastapi==0.119.1 uvicorn==0.38.0 python-dotenv==1.2.1
    fi
fi

[ "$DRY_RUN" = 1 ] || ensure_psi0_dotenv
echo "== done"
