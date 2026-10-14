# Policy evaluation

The paper's results are real-robot trials (10 per cell): end-task success rate
(TSR) in the main text and stage-wise cumulative success in App. B. This
directory evaluates the same policies offline and in simulation:

| Script | Purpose |
|--------|---------|
| `openloop_eval.py` | Replay LeRobot episodes through a policy server; per-dim action error |
| `openloop_plot.py` | Figures from the CSVs written by `openloop_eval.py` |
| `score_sim_episodes.py` | Task success and subtask stages of recorded sim rollouts |
| `plot_subtask_results.py` | Stage-wise cumulative success plot (App. B, Fig. 6) from per-trial CSVs |

All three VLA backbones take the 18-D state `observation.state_psi0_18`, the
ego-view RGB image and the task prompt, and return chunks of the 18-D action
`action.psi0_18` (layout at the end of this file).

## 1. Serve a checkpoint

The servers live in the Psi0 submodule. Run them from its root, each in its own
environment (see SETUP.md):

| Backend | Command (from `submodules/Psi0`) | Protocol | Port | Executed chunk |
|---------|----------------------------------|----------|------|----------------|
| `psi0`  | `bash scripts/deploy/serve_psi0_simple.sh <run_dir> <ckpt_step>` | HTTP `/act` | 22085 | 30 |
| `pi05`  | `bash baselines/pi05/serve_pi05.sh <config> <ckpt_dir> [port]` | websocket | 9000 | 16 |
| `gr00t` | `bash baselines/gr00t-n1.6/deploy_gr00t_simple.sh <ckpt_dir> [port]` | HTTP `/act` | 5556 | 16 |

```bash
cd submodules/Psi0
# psi0: the run dir is .runs/finetune/<EXP>.real.flow1000.cosine.lr1.0e-04.b256.gpus<N>.<timestamp>
#       and holds run_config.json, argv.txt and checkpoints/ckpt_<step>
bash scripts/deploy/serve_psi0_simple.sh .runs/finetune/psi0-legs_turn_uni-head-20000.<...>.<timestamp> 20000
# pi0.5: <config> is the config of the dataset the run was trained on; stats are read from that dataset (see below)
bash baselines/pi05/serve_pi05.sh legs_turn_uni_pi05 \
    .runs/openpi-05/legs_turn_uni_pi05/pi05-legs_turn_uni-full-20000/20000
# GR00T N1.6: embodiment G1_EEF_UNI (the 18-D LEGS schema)
bash baselines/gr00t-n1.6/deploy_gr00t_simple.sh .runs/gr00t-n16/gr00t-n16-legs_turn_uni-eef-20000/checkpoint-20000
```

Each script lists its environment overrides at the top (for example `PORT`;
`CUDA_VISIBLE_DEVICES` for ψ0 and π0.5, `DEVICE` for GR00T) and prints its usage
when run without arguments.

The π0.5 server does not read normalization stats from the checkpoint. It loads
`norm_stats.json` from the dataset of `<config>`, at
`$HF_LEROBOT_HOME/hojjunekim/<dataset>/`. `serve_pi05.sh` defaults `HF_LEROBOT_HOME`
to `$LEGS_CACHE/lerobot` (`~/.cache/legs/lerobot`), as the training scripts do; if
you trained with another value, serve with the same one. Keep that dataset in place.

## 2. Closed loop in simulation

Generate the scene XML for the scene you evaluate, then start the backend (the
MuJoCo sim loop and the SONIC controller) as in
[SETUP.md, section 5](../../SETUP.md#5-run-the-simulator). The backend reads the
XML only when it starts, so regenerate it and restart the backend when you switch
between `orange_to_plate_locomani` and `orange_to_plate_turn`. Then start the
frontend in the `legs` env, pointed at a running server. From the repository
root:

```bash
source ./init.sh
(cd scripts/main && python scene_config.py orange_to_plate_turn)   # then (re)start the backend
cd scripts/main
python simulate_object.py --scene orange_to_plate_turn \
    --policy_host localhost --backend psi0 --out_dir policy_turn_psi0
```

- `--backend {psi0,pi05,gr00t}` (default `psi0`) selects the wire protocol, the
  executed chunk length (30 / 16 / 16) and the default port; `--policy_port`
  overrides the port.
- The `pi05` backend (here and in `openloop_eval.py`) needs `openpi-client` in
  the `legs` env: `pip install -e submodules/Psi0/src/openpi/openpi-client`
  ([SETUP.md, section 8](../../SETUP.md#8-training-environments)).
- The prompt defaults to the scene's `prompt` in `scene_config.SCENES`;
  `--policy_prompt` overrides it.
- In the viser UI (`http://localhost:8080`), open **Policy Control** and click
  **Toggle Policy**. **Toggle RTC / Sequential** switches between asynchronous
  inference that blends overlapping chunks (RTC) and synchronous inference that
  executes each chunk in full.
- To score a rollout, click **🔴 Toggle Recording** before enabling the policy
  and again once the object has come to rest. Episodes are written as
  `episode_XXXX.json` plus `episode_XXXX_scenario.json`. A bare `--out_dir` name
  is a prefix: each simulator launch writes to
  `data/collected/<out_dir>_<timestamp>/` under the repository root. A path
  (absolute, or containing `/`) is used as the directory itself, and a relative
  one is taken from `scripts/main`.
- Between trials, click **Toggle Policy** to stop the policy, then
  **Reset MuJoCo Sim** (in **Interactive Object**). It resets the scene and turns
  the robot to a random yaw, which takes about 4 s. Then start the next
  recording.

## 3. Score sim rollouts

Run sections 3 to 5 from the repository root, in the `legs` env.
`score_sim_episodes.py` takes episode directories or `episode_XXXX.json` files.
To score every launch of the section 2 command:

```bash
python scripts/eval/score_sim_episodes.py data/collected/policy_turn_psi0_*/ \
    --backbone psi0 --condition "LEGS (200)" --out psi0_legs200.csv
```

Stages follow each scene's `task_plan` and are scored from object, plate and
robot poses only (recorded gripper commands are not used):

| Task | Scenes | Stages |
|------|--------|--------|
| 2, loco-manipulation | `orange_to_plate_locomani` | walk, pick, place |
| 3, long-horizon | `orange_to_plate_turn` | walk, pick, turn, walk, place (squat) |

Scene variants (those with `base_scene` set) are reported under their base
scene's task. Each stage is searched for from the frame where the previous one
was met:

- **walk**: the base comes within 0.45 m (xy) of the pick object, or of the
  place target while holding the object;
- **pick**: the object stays at least 3 cm above its start, within 0.25 m of a
  wrist and within 0.10 m of that wrist's height, for 0.3 s;
- **turn**: while holding, the heading is within 30° of the bearing to the target;
- **place**: over the last 0.3 s of the recording, the object rests on the
  target at constant height (it may still roll).

All thresholds are in `TOLERANCES` at the top of the script. Release is not
checked, since wrist-to-object distance does not separate held from released.
The script prints per-task success and cumulative stage percentages, and
`--out` writes the per-trial CSV described below. Pass `--scene` for recordings
made without `--scene`. On the scripted Task 2 and Task 3 demonstrations, every
episode scores as a success (200/200 each).

## 4. Open-loop evaluation

With a server running:

```bash
python scripts/eval/openloop_eval.py --backend psi0 \
    --dataset-path <lerobot_dataset> --episode 0-9 --stride 4
python scripts/eval/openloop_plot.py \
    --save-dir scripts/eval/eval_results/<dataset_name>/psi0
```

Each sampled frame's image, 18-D state and task prompt (from
`meta/tasks.jsonl`, or `--prompt`) are sent with `reset=True`, so every sample
is independent. The predicted chunk is compared with the next recorded actions.
For each dim the script reports the first-step L1 error and the L1 error
averaged over the executed chunk (`--horizon` overrides it). Wrist roll/pitch/yaw
errors are wrapped to ±180°.

Results go to `scripts/eval/eval_results/<dataset_name>/<backend>/`: per-episode
CSVs (`ep{N}_gt`, `_first_step`, `_horizon_avg`, `_chunks`, `_timings`), plus
`ep{N}_meta.json` and a `summary.json` with the mean per-dim errors. The dataset
is read directly in the LeRobot v2.1 layout (parquet + mp4), so `lerobot` is not
needed. The image, state and action columns can be changed with `--image-key`,
`--state-key` and `--action-key`.

## 5. Stage-wise plots (App. B)

```bash
python scripts/eval/plot_subtask_results.py psi0.csv pi05.csv gr00t.csv \
    --output stagewise_success.pdf
```

The script draws one panel per (backbone, task), with one line per data
condition giving the percentage of trials still successful after each stage.
The last point is the TSR. It also prints the same numbers as a table.

Per-trial CSV, one row per trial (`score_sim_episodes.py --out` writes it; for
real-robot trials, write one row per trial by hand):

| Column | Required | Meaning |
|--------|----------|---------|
| `task` | yes | `locomani` or `turn` (any other name needs `stages`) |
| `backbone` | yes | `psi0`, `pi05` or `gr00t` (other labels are plotted after these) |
| `condition` | yes | e.g. `LEGS (200)`, `LEGS (50)`, `SAM3D (200)`, `Teleop (50)` |
| `trial` | yes | trial index within its (task, backbone, condition) cell |
| `stages_completed` | yes | stages completed in order, 0 to the number of stages |
| `stages` | no | stage names joined by `\|`; defaults to the paper's stages above |
| `success` | no | 1 if every stage was completed (checked against `stages_completed`) |
| `episode` | no | recording the trial was scored from |

```csv
task,backbone,condition,trial,stages_completed,stages,success
turn,psi0,LEGS (200),0,5,walk|pick|turn|walk|place,1
turn,psi0,LEGS (200),1,2,walk|pick|turn|walk|place,0
```

## 18-D state and action

| Index | Component | Unit |
|-------|-----------|------|
| 0-2   | Left wrist position | m |
| 3-5   | Left wrist roll, pitch, yaw | rad |
| 6     | Left gripper | 0.5 open, 1.0 closed |
| 7-9   | Right wrist position | m |
| 10-12 | Right wrist roll, pitch, yaw | rad |
| 13    | Right gripper | 0.5 open, 1.0 closed |
| 14-16 | Base velocity command vx, vy, vyaw | m/s, rad/s |
| 17    | Height command | m |

The state uses the same layout. Indices 0-13 hold the current wrist and
gripper targets, 14-16 the previous velocity command (zero at the start), and
17 the height command.
