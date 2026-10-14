#!/usr/bin/env bash
# End-to-end LEGS data collection: record episodes in sim -> render them
# offline over a Gaussian splat -> build the universal LeRobot v2.1 dataset.
#
# Usage:  ./run_collection.sh [N_EPISODES] [DATASET_NAME] [SCENE]
#   defaults: N=1, NAME=g1_legs_test, SCENE=orange_to_plate_locomani
#   paper scenes: orange_to_plate_locomani (T2), orange_to_plate_turn (T3)
#
# Env knobs:
#   STOP_AFTER=N    stop after step N (5, 6, or 7). Default 7 = full pipeline.
#                     5 = stop after raw episodes (no snapshot/render/build)
#                     6 = stop after CSV snapshot (raw episodes + token CSVs, no render/build)
#                     7 = full pipeline incl. universal LeRobot dataset
#   PLY=<name>      render background for step 7 (data/gsplat/<name>.ply).
#                     Default wood, the background of the paper's LEGS datasets
#                     (fetch it with scripts/download_assets.sh paper); flightroom comes
#                     with the core assets (scripts/download_assets.sh core).
#   GROOT_WBC_DIR   GR00T-WholeBodyControl checkout (default: the submodule).
#                     scene_config.py writes generated_scene.xml into it, so the
#                     deploy and sim loop below must run from the same checkout.
#   CONDA_BASE      conda install root (default: `conda info --base`).
#
# Prereqs (manual, see SETUP.md section 5):
#   1. Before running this script, open the docker container in Shell B
#      ($GROOT_WBC_DIR/gear_sonic_deploy/docker/run-ros2-dev.sh, container `g1-deploy-dev`).
#   2. When the script asks, start the backend in this order:
#        Shell A, on the host, from $GROOT_WBC_DIR, in a new shell without the legs env:
#          source /opt/ros/humble/setup.bash && source .venv_sim/bin/activate
#          python gear_sonic/scripts/run_sim_loop.py
#        Shell B, inside docker, once the MuJoCo window is open:
#          cd /workspace/g1_deploy
#          EXTRA_DEPLOY_ARGS="--enable-csv-logs --logs-dir collect_logs" \
#              bash deploy.sh sim --input-type manager
#      When the deploy prints "Planner initialized successfully!", click the
#      MuJoCo window and press 9 to release the elastic band.
#   3. Once this script launches viser, open http://localhost:8080 and click
#      "🔴 Auto Record". Recording stops on its own after N successful episodes.
#   The frontend and backend talk over ROS 2 / DDS on domain 0, so run one
#   collection stack per machine.
#
# Deploy CSVs (incl. token_state.csv) are snapshotted at recording end and
# passed to build_sonic_lerobot.py via --token-csv-dir so action.motion_token
# is populated. Only SONIC fine-tuning reads that column; psi0 / pi05 /
# GR00T N1.6 train on the 18-D EEF columns, so the CSV logs are optional.

set -eo pipefail

# If a Python venv is active in the parent shell (e.g. .venv-psi), its bin/
# stays at the front of PATH and silently shadows `conda activate legs` in
# the subshells below. Strip it now so legs actually wins.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "^${VIRTUAL_ENV}/bin$" | paste -sd:)
    unset VIRTUAL_ENV
fi

N=${1:-1}
NAME=${2:-g1_legs_test}
SCENE=${3:-orange_to_plate_locomani}
STOP_AFTER=${STOP_AFTER:-7}
PLY=${PLY:-wood}
if [[ "$STOP_AFTER" =~ ^[0-9]+$ ]] && (( STOP_AFTER >= 5 && STOP_AFTER <= 7 )); then
    :
else
    echo "ERROR: STOP_AFTER must be 5, 6, or 7 (got '$STOP_AFTER')" >&2
    exit 1
fi
# Timestamped data dir so multiple runs don't collide. Lets you start a new
# recording while a previous render/build is still in flight on its own dir.
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
DATA_DIR=$PROJECT/data/collected/${NAME}_${TIMESTAMP}
SAVE_ROOT=$PROJECT/data/lerobot/wigs/${NAME}_uni
DOCKER=g1-deploy-dev
GROOT_WBC_DIR=${GROOT_WBC_DIR:-$PROJECT/submodules/GR00T-WholeBodyControl}
export GROOT_WBC_DIR
DEPLOY=$GROOT_WBC_DIR/gear_sonic_deploy
DATA_PY="${LEGS_DATA_PY:-$PROJECT/submodules/Psi0/.venv-openpi/bin/python}"
SIM_LOG=$DATA_DIR/simulate_object.log
LAST_EP=$DATA_DIR/episode_$(printf '%04d' $((N-1))).json
CSV_SNAPSHOT=$DATA_DIR/csv_snapshot

echo "============================================================"
echo " LEGS collection: N=$N episodes, scene=$SCENE"
echo "   data dir: $DATA_DIR"
echo "   dataset : $SAVE_ROOT"
echo "============================================================"

# ---- pre-flight ----
[[ -f "$GROOT_WBC_DIR/gear_sonic/scripts/run_sim_loop.py" ]] \
    || { echo "ERROR: no GR00T-WholeBodyControl at $GROOT_WBC_DIR (check out the submodules: SETUP.md, section 1)"; exit 1; }
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] \
    || { echo "ERROR: conda not found at $CONDA_BASE (set CONDA_BASE)"; exit 1; }
CORE_MISSING=$(cd "$PROJECT" && awk '$1 == "core" {print $4}' scripts/assets.sha256 \
    | while read -r f; do [[ -f "$f" ]] || echo "$f"; done)
[[ -z "$CORE_MISSING" ]] \
    || { echo "ERROR: core assets missing, e.g. ${CORE_MISSING%%$'\n'*} (run scripts/download_assets.sh core)"; exit 1; }
if (( STOP_AFTER >= 7 )); then
    # Same resolution as offline_renderer._resolve_ply_path (cwd scripts/data).
    if [[ "$PLY" == */* ]]; then
        PLY_FILE=$(cd "$PROJECT/scripts/data" && realpath -m "$PLY")
    else
        PLY_FILE=$PROJECT/data/gsplat/${PLY%.ply}.ply
    fi
    if [[ ! -f "$PLY_FILE" ]]; then
        GROUP=$(awk -v p="${PLY_FILE#"$PROJECT"/}" '$1 !~ /^#/ && $4 == p {print $1}' "$PROJECT/scripts/assets.sha256")
        echo "ERROR: $PLY_FILE missing (run scripts/download_assets.sh ${GROUP:-paper}, or set PLY=flightroom)"
        exit 1
    fi
    # The renderer silently falls back to an uncalibrated splat without this.
    [[ -f "${PLY_FILE%.ply}/default.json" ]] \
        || { echo "ERROR: ${PLY_FILE%.ply}/default.json (splat-to-world calibration) missing"; exit 1; }
    [[ -x "$DATA_PY" ]] \
        || { echo "ERROR: dataset-builder python $DATA_PY missing (see SETUP.md, section 7; or set LEGS_DATA_PY)"; exit 1; }
fi
docker ps --format '{{.Names}}' | grep -q "^${DOCKER}$" \
    || { echo "ERROR: docker container '$DOCKER' not running (open it in Shell B with run-ros2-dev.sh, see SETUP.md section 5)"; exit 1; }

echo "[1/7] Creating fresh data dir: $DATA_DIR"
mkdir -p "$DATA_DIR"

echo "[2/7] Regenerating MuJoCo scene XML for '$SCENE'..."
(
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate legs
  cd "$PROJECT/scripts/main"
  python scene_config.py "$SCENE"
)

cat <<EOF

Now (re)start the backend so it loads the new scene, in this order:
  shell A (host, in $GROOT_WBC_DIR, a new shell without the legs env):
    source /opt/ros/humble/setup.bash
    source .venv_sim/bin/activate
    python gear_sonic/scripts/run_sim_loop.py

  shell B (inside docker, /workspace/g1_deploy), once the MuJoCo window is open:
    EXTRA_DEPLOY_ARGS="--enable-csv-logs --logs-dir collect_logs" \\
    bash deploy.sh sim --input-type manager
  and press Enter at its "Proceed with deployment? [Y/n]" prompt.

  When shell B prints "Planner initialized successfully!", click the MuJoCo
  window and press 9 to release the elastic band.

EOF
read -r -p "Press ENTER once the robot stands on its own... " _

# ---- launch sim ----
echo "[3/7] Launching simulate_object.py (logs -> $SIM_LOG)..."
START_TIME=$(date +%s)
(
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  cd "$PROJECT"
  source ./init.sh
  cd "$PROJECT/scripts/main"
  # --ply here only sets the viewer background and the walking planner's
  # occupancy grid. Keep it at flightroom: the paper's recordings were planned
  # on the flightroom grid. The render background is chosen in step 7 ($PLY).
  # exec, so $SIM_PID below is the python process itself. -u, so its
  # [AutoRecord] lines reach this terminal as they happen: it is SIGKILLed at
  # the end, which would drop output still in its buffer.
  exec python -u simulate_object.py --scene "$SCENE" --ply flightroom --out_dir "$DATA_DIR"
) >"$SIM_LOG" 2>&1 &
SIM_PID=$!
sleep 4
echo "    sim PID=$SIM_PID. Open viser in browser and click the auto-record button."
echo "    (On the first start, gsplat compiles its CUDA kernels at the first render, about"
echo "    1.5 min; viser can look stuck until then.)"
echo "    Recording will end automatically after $N successful episode(s)."

# Stream per-episode progress + success/fail from sim log to this terminal.
# auto_recorder.py prefixes those lines with [AutoRecord]; everything else
# stays in $SIM_LOG.
stdbuf -oL tail -F -n 0 "$SIM_LOG" 2>/dev/null \
    > >(stdbuf -oL grep --line-buffered -E "^\[AutoRecord\]" | sed -u 's/^/    /') &
TAIL_PID=$!

# ---- watcher ----
echo "[4/7] Watching for $LAST_EP ..."
echo "    *** Open http://localhost:8080 and click '🔴 Auto Record' button ***"
while [[ ! -f "$LAST_EP" ]]; do
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
        echo "ERROR: simulate_object.py died early. See $SIM_LOG"
        exit 1
    fi
    sleep 5
done
# wait for size to stabilize
prev=-1; cur=$(stat -c%s "$LAST_EP")
while [[ "$prev" != "$cur" ]]; do sleep 5; prev=$cur; cur=$(stat -c%s "$LAST_EP"); done
# Stop streaming sim log to terminal.
kill "$TAIL_PID" 2>/dev/null || true
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf "    Done. %d episodes in %dh %02dm %02ds (%ds total).\n" \
    "$N" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"

# ---- shutdown ----
echo "[5/7] Stopping simulator (deploy + sim_loop you'll stop manually)..."
# SIGKILL immediately — SIGINT grace lets the auto-record state machine
# walk into the next episode and save episode_$((N)).json before death.
kill -SIGKILL "$SIM_PID" 2>/dev/null || true
wait "$SIM_PID" 2>/dev/null || true

# Drop any over-recorded episodes (rare race: a save flushed between
# watcher detection and kill). Keep only episode_0000..episode_$((N-1)).
for f in "$DATA_DIR"/episode_*.json; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f")
    # Strip "episode_" prefix and ".json"/"_scenario.json" suffix
    idx=${base#episode_}; idx=${idx%.json}; idx=${idx%_scenario}
    if (( 10#$idx >= N )); then
        rm -f "$f" "${f%.json}_scenario.json"
        echo "    pruned $base (idx $idx >= N=$N)"
    fi
done

if (( STOP_AFTER <= 5 )); then
    echo "============================================================"
    echo " STOP_AFTER=$STOP_AFTER — stopping after step 5 (raw episodes)."
    echo "   Episodes  : $N"
    echo "   Data dir  : $DATA_DIR"
    echo "   Sim log   : $SIM_LOG"
    echo "   Wall time : $(printf '%dh %02dm %02ds' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))"
    echo "============================================================"
    exit 0
fi

# ---- snapshot deploy CSVs ----
# Captured immediately after recording so a wiped/rotated collect_logs later
# (or a new deploy run) can't mask the data. build_sonic_lerobot reads
# token_state.csv from here via --token-csv-dir — without it,
# action.motion_token is all zeros and the dataset is unusable for SONIC.
echo "[6/7] Snapshotting deploy CSVs to $CSV_SNAPSHOT ..."
mkdir -p "$CSV_SNAPSHOT"
csv_n=$(ls -1 "$DEPLOY"/collect_logs/*.csv 2>/dev/null | wc -l)
TOKEN_CSV_DIR=""
if (( csv_n > 0 )); then
    cp "$DEPLOY"/collect_logs/*.csv "$CSV_SNAPSHOT"/
    TOKEN_CSV_DIR=$CSV_SNAPSHOT
    echo "    snapshot: copied $csv_n CSV(s) (incl. token_state.csv if present)"
else
    echo "    NOTE: no CSVs in $DEPLOY/collect_logs (deploy ran without"
    echo "          EXTRA_DEPLOY_ARGS='--enable-csv-logs --logs-dir collect_logs')."
    echo "          action.motion_token will be zero-filled; only SONIC fine-tuning reads it."
fi

if (( STOP_AFTER <= 6 )); then
    echo "============================================================"
    echo " STOP_AFTER=$STOP_AFTER — stopping after step 6 (raw + CSV snapshot)."
    echo "   Episodes  : $N"
    echo "   Data dir  : $DATA_DIR"
    echo "   CSV snap  : $CSV_SNAPSHOT"
    echo "   Sim log   : $SIM_LOG"
    echo "   Wall time : $(printf '%dh %02dm %02ds' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))"
    echo "============================================================"
    exit 0
fi

# ---- render + build ----
echo "[7/7] Rendering + building universal LeRobot dataset (ply=$PLY)..."
# N=5 workers: ~3.75× speedup vs single-process on RTX-class GPU (~119 ms/frame
# under contention, ~770 MiB per worker). MP script auto-clamps workers to
# min(--workers, len(episodes)), so passing 5 with <5 episodes is safe.
(
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate legs
  cd "$PROJECT/scripts/data"
  python offline_renderer_mp.py --data-dir "$DATA_DIR" --ply "$PLY" --workers 5
) 2>&1 | tail -15

# wigs2universal: build_sonic_lerobot -> add_18state_column -> ... -> stats.
# The output (observation.images.ego_view, observation.state_psi0_18,
# action.psi0_18) trains psi0, pi0.5 and GR00T N1.6 via scripts/train/.
# Third arg = token-csv-dir (snapshotted above) so action.motion_token is real.
LEGS_DATA_PY=$DATA_PY "$PROJECT/scripts/data/wigs2universal.sh" "$DATA_DIR" "$SAVE_ROOT" "$TOKEN_CSV_DIR"

echo "============================================================"
echo " Collection complete."
echo "   Episodes : $N"
echo "   Wall time: $(printf '%dh %02dm %02ds' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))"
echo "   Dataset  : $SAVE_ROOT"
echo "   Sim log  : $SIM_LOG"
echo "============================================================"
