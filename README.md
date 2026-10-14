# LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World

**CoRL 2026** · [Project page](https://hojjunekim.github.io/legs/) · [Paper](https://arxiv.org/abs/2606.01458)

LEGS generates humanoid training data without teleoperation. A procedural motion generator drives a
Unitree G1 in MuJoCo and records each episode. Afterwards, every episode is rendered offline by
compositing the robot and object meshes over a 3D Gaussian splat of the real scene. A color
calibration then maps the result to the robot's RealSense camera. The episodes are exported as
LeRobot v2.1 datasets and used to fine-tune ψ0, π0.5 and GR00T N1.6.

| Stage | Code |
|---|---|
| Scenes and tasks | `scripts/main/scene_config.py` |
| Simulation and recording | `scripts/main/simulate_object.py`, `run_collection.sh` |
| Gaussian-splat capture | `scripts/gsplat/video2gsplat.py` |
| Offline rendering | `scripts/data/offline_renderer*.py` |
| Color calibration | `scripts/calibration/`, `scripts/main/object_gen/` |
| Dataset export | `scripts/data/*2universal.sh` |
| Training (ψ0, π0.5, GR00T N1.6) | `scripts/train/` |
| Evaluation | `scripts/eval/` |

Physics and whole-body control run in GR00T-WholeBodyControl (MuJoCo plus the SONIC controller),
included as a submodule. The LEGS frontend sends it an 18-D wrist, gripper and base command and
receives the robot state over ROS 2 / DDS.

## Tasks

| Task | Scene | Prompt | Training set (Hugging Face) |
|---|---|---|---|
| 2, loco-manipulation | `orange_to_plate_locomani` | place the orange on the plate | [`hojjunekim/legs_locomani_uni`](https://huggingface.co/datasets/hojjunekim/legs_locomani_uni) |
| 3, long-horizon | `orange_to_plate_turn` | pick the orange, turn right, and place it on the plate | [`hojjunekim/legs_turn_uni`](https://huggingface.co/datasets/hojjunekim/legs_turn_uni) |

The training sets have 200 episodes each, rendered over the wood-table splat. The sets of the
paper's other data conditions are listed under [Paper datasets](#paper-datasets).

## Quick start

See [`SETUP.md`](SETUP.md) to install the environments and the simulation backend. Then:

```bash
bash scripts/download_assets.sh core paper   # meshes and splats, about 315 MB to download
source ./init.sh
(cd scripts/main && python scene_config.py orange_to_plate_locomani)   # writes the scene XML the backend loads
```

Start the MuJoCo sim loop and the SONIC controller in two more shells and release the elastic band
([SETUP.md, section 5](SETUP.md#5-run-the-simulator)). Then start the frontend and open
`http://localhost:8080`:

```bash
cd scripts/main && python simulate_object.py --scene orange_to_plate_locomani
```

For Task 3, pass `orange_to_plate_turn` to both commands. The backend loads the scene XML when it
starts, so after a scene change rerun `scene_config.py` and restart the backend.

To record, render and export a dataset in one go, run `./run_collection.sh <N_EPISODES> <NAME> <SCENE>`
([SETUP.md, section 6](SETUP.md#6-end-to-end-collection)).

## Layout

| Path | Purpose |
|---|---|
| `scripts/main/` | Simulator (`simulate_object.py`), scene config, controllers, task primitives, renderers |
| `scripts/main/object_gen/` | SAM 3D object meshes and their color refinement |
| `scripts/data/` | Offline rendering and LeRobot dataset export |
| `scripts/gsplat/` | Video → Gaussian splat (nerfstudio) |
| `scripts/calibration/` | iPhone → RealSense color matrix |
| `scripts/train/` | Training scripts for ψ0, π0.5 and GR00T N1.6 |
| `scripts/eval/` | Scoring of sim rollouts, open-loop evaluation, stage-wise plots (the policy servers are in `submodules/Psi0`) |
| `scripts/download_assets.sh`, `scripts/assets.sha256` | Large assets that are not in git |
| `run_collection.sh` | End-to-end collection: record → render → dataset |
| `data/objects/`, `data/gsplat/`, `robots/g1/` | Meshes (with the paper's collision parts in `data/objects/<name>/`), splats and the G1 model, fetched by `scripts/download_assets.sh`; recordings and datasets are written under `data/` |
| `submodules/` | GR00T-WholeBodyControl (backend), Psi0 (training and serving), sam-3d-objects |
| `legs.yml`, `gsplat.yml` | Conda environments for the simulator and for splat capture |

## Large assets

Meshes and splats are not stored in git. `scripts/download_assets.sh` fetches them to their repo paths:

```bash
bash scripts/download_assets.sh core      # required: G1, object and table meshes, flightroom splat (186.5 MB archive)
bash scripts/download_assets.sh paper     # wood-table splat + calibration, HDR-refit orange and plate (128.0 MB)
bash scripts/download_assets.sh aug       # appearance study: blue-table splat + calibration, apple, box (146.4 MB)
bash scripts/download_assets.sh sources   # raw SAM 3D meshes: mesh-only baseline, object_gen inputs (66.8 MB)
bash scripts/download_assets.sh all       # or --list to see files and what is present
```

`core` is one archive on Google Drive, [`LEGS-core-assets.tar.gz`](https://drive.google.com/file/d/15lTnNmjF-sB8RUzDTdFPacZu8WVzstv6/view) (284.2 MB unpacked); the other groups
are files in the Hugging Face dataset [`hojjunekim/LEGS-assets`](https://huggingface.co/datasets/hojjunekim/LEGS-assets).
Every file is checked against `scripts/assets.sha256`, and re-runs skip intact files. The script needs
curl, tar and sha256sum (or shasum).

Google Drive limits downloads of a widely shared file for up to a day. If the script cannot get the
archive, it prints the archive's Drive link: download it in a browser, then run
`bash scripts/download_assets.sh core --archive <file>`. To change the sources, set `LEGS_CORE_ARCHIVE_URL`
(the archive), or `LEGS_ASSETS_REPO`, `LEGS_ASSETS_REVISION` or `LEGS_ASSETS_BASE_URL` (a mirror of the
Hugging Face files), and set `HF_TOKEN` for a private copy.

The flightroom splat (`data/gsplat/flightroom.ply`) comes with `core`; its pose in the scene,
`flightroom/default.json`, is in git. The simulator shows it by default and plans walking paths on its
occupancy grid, as the paper's recordings did. The paper's training images are rendered over the wood
splat, so the renderer needs `paper`. Without it, pass `--ply flightroom`. If the HDR-refit meshes are
missing, the renderer warns and falls back to `orange.glb` / `plate.glb`, whose colors do not match the
paper.

## Collection

`scripts/main/simulate_object.py` runs the viser frontend (`http://localhost:8080`) against the
GR00T-WholeBodyControl backend at 30 Hz. **🔴 Auto Record** generates and records episodes until as many
as set in **Episodes to record** have succeeded. Each episode is written as `episode_XXXX.json`
(per-frame robot state, camera and object poses, and the wrist, gripper and base commands) with
`episode_XXXX_scenario.json` (scene layout, camera metadata and prompt).

`run_collection.sh` chains the whole flow:

1. regenerate the scene XML;
2. record N episodes;
3. snapshot the deploy CSVs (SONIC motion tokens);
4. render;
5. build the LeRobot dataset under `data/lerobot/wigs/<NAME>_uni` (`wigs` is the project's earlier name).

The backend runs in two extra shells, as set out in SETUP.md. `STOP_AFTER=5` stops after recording and
`STOP_AFTER=6` after the CSV snapshot. These are the script's own step numbers (`[5/7]` and `[6/7]`); the
default, 7, runs everything. `PLY=<name>` picks the render splat (default `wood`).

## Rendering

```bash
cd scripts/data
python offline_renderer_mp.py --data-dir <raw-dir> --workers 3
```

`<raw-dir>` is a recording directory, for example `../../data/collected/<NAME>_<timestamp>/` from
`scripts/data`, or an absolute path. Per frame, the renderer composites the robot and object meshes over
the splat by depth; the tables in the images are the splat's own. It then applies the iPhone → RealSense
color matrix and writes JPEGs to `<raw-dir>/final_renders/`.

- `--ply` defaults to the scene's splat (a variant's own, otherwise `wood`); a bare name means
  `data/gsplat/<name>.ply`.
- A splat without `data/gsplat/<name>/default.json` is an error unless you pass
  `--allow-uncalibrated` (identity pose).
- For quick checks, set `MAX_FRAMES=1` or `FRAME_IDX=N` (both imply `SKIP_AUG=1`); `OUT_SUBDIR=<dir>`
  writes to `<raw-dir>/<dir>/` (or to `<dir>` if it is absolute) instead of `<raw-dir>/final_renders/`.

### Appearance study

Recorded episodes re-render under a new scene appearance without re-simulating (needs `aug`). `--scene`
must be a variant of the episode's recorded scene. Variants are render-only: `scene_config.py` refuses
to build their scene XML, so record with the base scene and re-render.

```bash
python offline_renderer_mp.py --data-dir <raw-dir> --scene apple_to_box_turn_blue --workers 3
```

| `--scene` | Recorded scene | Objects | Tables / splat |
|---|---|---|---|
| `orange_to_plate_turn_blue` | `orange_to_plate_turn` | orange, plate | blue high + wood mid / blue |
| `orange_to_plate_locomani_blue` | `orange_to_plate_locomani` | orange, plate | blue high / blue |
| `apple_to_box_turn` | `orange_to_plate_turn` | apple, box | default / wood |
| `apple_to_box_locomani` | `orange_to_plate_locomani` | apple, box | default / wood |
| `apple_to_box_turn_blue` | `orange_to_plate_turn` | apple, box | blue high + wood mid / blue |
| `apple_to_box_locomani_blue` | `orange_to_plate_locomani` | apple, box | blue high / blue |

When exporting a variant, pass its prompt as the `[task]` argument of `wigs2universal.sh`, because the
scenario files keep the recorded orange/plate prompt.

### Mesh-only baseline

This is the paper's SAM3D condition: SAM 3D meshes with no splat.

```bash
bash scripts/download_assets.sh sources
cd scripts/data
python offline_renderer_baseline_mp.py --data-dir <raw-dir> [--scene apple_to_box_turn] --workers 3
```

The renders go to `<raw-dir>/baseline_renders/`; export them with `sam2universal.sh`.

### Capturing a new splat

```bash
conda env create -f gsplat.yml && conda activate legs-gsplat
python scripts/gsplat/video2gsplat.py <name> --video <capture.mov>
```

Any video that ffmpeg can read works. The script samples 300 frames uniformly (`num_images` in
`scripts/gsplat/iphone16e.json`, the default `--config`, written for 1280×720 iPhone 16e video), and SfM
estimates the camera intrinsics. Everything goes under `scripts/gsplat/output/<name>/` (`images/`,
`sfm/`, `nerfstudio/`, `export/`). A rerun reuses the frames and SfM results already there, so delete
that directory before re-running a name with a new video. `video2gsplat.py` runs SfM with hloc using
SuperPoint and SuperGlue. Their pretrained weights are downloaded at run time and are licensed for
non-commercial research use only.

The splat is written to `scripts/gsplat/output/<name>/export/splat.ply`; copy it to
`data/gsplat/<name>.ply`. SfM gives it no metric scale and an arbitrary origin, so place it in the world
before using it. The simulator runs in the `legs` env, so use a new shell:

```bash
source ./init.sh
cd scripts/main && python simulate_object.py --ply <name>   # add --scene orange_to_plate_turn to also show mid_table
```

Open `http://localhost:8080` and tick **Rendering → Show Static Meshes**. In **GSplat Calibration**, set
**Scale**, **X/Y/Z** and **Roll/Pitch/Yaw** (or tick **Show Gizmo** and drag) until the scanned floor is
at z = 0 and the scanned tables cover the simulated ones (`high_table` has its top at z = 0.730 m and is
centred at x = 1.0, y = 0). The renders do not draw the table meshes, so the tables in the images are the
splat's own. **Save Calibration** writes `data/gsplat/<name>/default.json` (`translation` in meters,
`rotation_rpy_deg` and `scale`; a splat point p lands at translation + R(rpy) · scale · p).

Then render over it with `--ply <name>` (`offline_renderer_mp.py`, from `scripts/data`) or with
`PLY=<name> ./run_collection.sh <N_EPISODES> <NAME> <SCENE>`; recording still plans walking paths on the
flightroom splat. Do not copy nerfstudio's `dataparser_transforms.json` into `data/gsplat/`: the
simulator and the renderer would apply it to every splat in that directory.

## Color calibration

The renderer applies a 3×3 iPhone → RealSense color matrix to each composited frame. The matrix is fit
from one ColorChecker photo pair in `data/calibration/color_chart/`. To check or refit it, run
`python scripts/calibration/fit_color_matrix.py` (see [`scripts/calibration/README.md`](scripts/calibration/README.md)).

The object meshes are color-refined to the iPhone domain. To reproduce `orange.glb` and `plate.glb` from
their inputs in `data/objects/sources/<name>/`, run `bash scripts/download_assets.sh sources`, then

```bash
python scripts/main/object_gen/run_pipeline.py --source data/objects/sources/orange --steps refine,render
python scripts/main/object_gen/run_pipeline.py --source data/objects/sources/plate --steps refine,render \
    --apply-ccm data/objects/sources/plate/ccm.json
```

The results go to `data/objects/sam3d_runs/<name>/` (see
[`scripts/main/object_gen/README.md`](scripts/main/object_gen/README.md), which also covers new objects).
The HDR-refit `orange_hdr.glb` and `plate_hdr.glb` that the paper's renders use are not rebuilt by this
pipeline; they come with `paper`.

## Dataset export

`scripts/data/wigs2universal.sh` (splat renders) and `sam2universal.sh` (mesh-only renders) convert a
directory of recorded episodes into a LeRobot v2.1 dataset at 30 fps. The dataset holds the ego-view
video `observation.images.ego_view` (480×640), the 18-D state and action used by all three policies,
and the normalization stats and modality files the trainers read. With the deploy CSVs it also fills
the SONIC motion tokens. SETUP.md, section 7, covers the environment and the arguments.

## Paper datasets

The training sets of every data condition in the paper's Tasks 2 and 3 are on Hugging Face under
`hojjunekim/`. All are LeRobot v2.1 datasets in the layout of [State and action](#state-and-action).

| Condition | Task 2, loco-manipulation | Task 3, long-horizon |
|---|---|---|
| Teleop (50) | [`teleop_locomani_uni`](https://huggingface.co/datasets/hojjunekim/teleop_locomani_uni) | [`teleop_turn_uni`](https://huggingface.co/datasets/hojjunekim/teleop_turn_uni) |
| SAM3D (50) | [`sam_locomani_50_uni`](https://huggingface.co/datasets/hojjunekim/sam_locomani_50_uni) | [`sam_turn_50_uni`](https://huggingface.co/datasets/hojjunekim/sam_turn_50_uni) |
| SAM3D (200) | [`sam_locomani_uni`](https://huggingface.co/datasets/hojjunekim/sam_locomani_uni) | [`sam_turn_uni`](https://huggingface.co/datasets/hojjunekim/sam_turn_uni) |
| SAM3D-AUG (200) | [`sam_locomani_aug_uni`](https://huggingface.co/datasets/hojjunekim/sam_locomani_aug_uni) | [`sam_turn_aug_uni`](https://huggingface.co/datasets/hojjunekim/sam_turn_aug_uni) |
| LEGS (50) | [`legs_locomani_50_uni`](https://huggingface.co/datasets/hojjunekim/legs_locomani_50_uni) | [`legs_turn_50_uni`](https://huggingface.co/datasets/hojjunekim/legs_turn_50_uni) |
| LEGS (200) | [`legs_locomani_uni`](https://huggingface.co/datasets/hojjunekim/legs_locomani_uni) | [`legs_turn_uni`](https://huggingface.co/datasets/hojjunekim/legs_turn_uni) |
| LEGS-AUG (50) | [`legs_locomani_50_aug_uni`](https://huggingface.co/datasets/hojjunekim/legs_locomani_50_aug_uni) | [`legs_turn_50_aug_uni`](https://huggingface.co/datasets/hojjunekim/legs_turn_50_aug_uni) |
| LEGS-AUG (200) | [`legs_locomani_aug_uni`](https://huggingface.co/datasets/hojjunekim/legs_locomani_aug_uni) | [`legs_turn_aug_uni`](https://huggingface.co/datasets/hojjunekim/legs_turn_aug_uni) |

- Teleop: real-robot demonstrations recorded with SONIC's VR teleoperation.
- SAM3D: the mesh-only baseline (SAM 3D meshes instead of the splat, no color calibration).
- LEGS: the robot and object meshes composited over the wood-table splat, with color calibration.
- AUG: the LEGS (200) episodes re-rendered for the [appearance study](#appearance-study), one look per
  quarter: episodes 0–49 as `apple_to_box_<task>_blue` (apple and box, blue table), 50–99 as
  `orange_to_plate_<task>_blue` (orange and plate, blue table), 100–149 unchanged, and 150–199 as
  `apple_to_box_<task>` (apple and box, original tables). The apple and box episodes carry the apple and
  box prompt.
- (50) and (200) are the numbers of episodes.

Within a task, every SAM3D and LEGS set is rendered from episodes of that task's LEGS (200) set, with
the same states and actions:

- SAM3D (50) and LEGS (50) hold episodes 0–49;
- LEGS-AUG (50) holds 12 or 13 episodes of each quarter of LEGS-AUG (200): episodes 0–11, 50–62,
  100–111 and 150–162;
- the exception is Task 2's SAM3D (200), which was rendered from 200 other episodes and shares none
  with LEGS (200).

## State and action

All three policies use the same 18-D layout for the state `observation.state_psi0_18` and the action
`action.psi0_18`:

| Index | Component | Unit |
|---|---|---|
| 0–2 | Left wrist position | m |
| 3–5 | Left wrist roll, pitch, yaw | rad |
| 6 | Left gripper | 0.5 open, 1.0 closed |
| 7–9 | Right wrist position | m |
| 10–12 | Right wrist roll, pitch, yaw | rad |
| 13 | Right gripper | 0.5 open, 1.0 closed |
| 14–16 | Base velocity command vx, vy, vyaw | m/s, rad/s |
| 17 | Height command | m |

In the state, indices 14–16 hold the previous step's velocity command (zero at the start of an episode).
The datasets also keep the older 15-D `observation.state_psi0`, which has no base velocity command.

## Training

The three policies from the paper (ψ0, π0.5 and GR00T N1.6) are fine-tuned with the scripts in
`scripts/train/` using the paper settings: 20k steps, with a global batch of 256, 128 and 24
respectively.

```bash
bash scripts/train/setup_envs.sh           # uv environments inside submodules/Psi0
bash scripts/train/download.sh             # LEGS datasets and base weights (default cache: ~/.cache/legs)
TASK=locomani bash scripts/train/psi0.sh   # or pi05.sh / groot.sh; TASK=turn for Task 3
```

The default datasets are the LEGS (200) sets. To train on another set from [Paper datasets](#paper-datasets),
fetch it by its id and pass the same id as `DATASET`:

```bash
bash scripts/train/download.sh --dataset hojjunekim/sam_turn_uni
TASK=turn DATASET=hojjunekim/sam_turn_uni bash scripts/train/psi0.sh
```

Add `--dry-run` to print the resolved configuration and command without starting a run. Weights & Biases
logging stays off unless `WANDB_PROJECT` is set. See [`scripts/train/README.md`](scripts/train/README.md)
for GPU counts, environment variables and output locations.

## Evaluation

[`scripts/eval/README.md`](scripts/eval/README.md) covers the following:

- serving a checkpoint of each policy;
- running it closed loop in the simulator (`simulate_object.py --policy_host localhost --backend {psi0,pi05,gr00t}`);
- scoring the recorded rollouts by task stage (`score_sim_episodes.py`);
- open-loop action error on a LeRobot dataset (`openloop_eval.py`);
- the stage-wise success plot of the paper's App. B.

The paper reports real-robot trials; these tools evaluate the same policies offline and in simulation.
The client that ran the policies on the real G1 is not part of this release.

## Other tools

- `python scripts/data/test_iphone_calib_render.py --data-dir <raw-dir> --episodes 5` writes one
  side-by-side MP4 per episode to `<raw-dir>/iphone_calib_test/`: the composite over the raw splat, and
  over the splat with the color matrix applied. Compare it with a RealSense recording of the same room.
- `scripts/data/view_scene_raw.py --scene <name>` shows a capture app's scan mesh next to the splat in
  viser and saves their alignment to `data/gsplat/<name>/mesh_to_gsplat.json`.
  `scripts/data/extract_scene_table.py --scene <name>` crops a table out of that mesh as a MuJoCo static
  mesh. Both need the scan's `data/gsplat/<name>/3DModel.obj`, which this release does not include.
- `scripts/main/object_gen/pick_clicks.py` picks the SAM 2 click prompts for a new object (see the
  [object_gen README](scripts/main/object_gen/README.md)). `sam2_video_mask.py` next to it writes a
  per-object SAM 2 mask video from clicks on the first frame; it runs in the `sam3d-objects` env.

## Acknowledgements

LEGS builds on the following projects:

- [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl): the MuJoCo sim loop and
  the SONIC whole-body controller. `submodules/GR00T-WholeBodyControl` is a fork of it.
- [Psi0](https://github.com/physical-superintelligence-lab/Psi0): ψ0 and the training and serving code
  of all three policies. `submodules/Psi0` is a fork of it.
- [openpi](https://github.com/Physical-Intelligence/openpi) (π0.5) and
  [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) (GR00T N1.6).
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) and
  [SAM 2](https://github.com/facebookresearch/sam2): object meshes and masks.
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio),
  [gsplat](https://github.com/nerfstudio-project/gsplat) and
  [hloc](https://github.com/cvg/Hierarchical-Localization): splat capture and rendering.
- [viser](https://github.com/viser-project/viser): the frontend.
- [MuJoCo](https://github.com/google-deepmind/mujoco), [CoACD](https://github.com/SarahWeiii/CoACD) and
  [LeRobot](https://github.com/huggingface/lerobot): physics, convex decomposition and the dataset format.
- [unitree_ros](https://github.com/unitreerobotics/unitree_ros): the Unitree G1 model.

## License

LEGS is released under the Apache License 2.0 (see `LICENSE`). The Unitree G1 model in `robots/g1`,
including the meshes that `core` adds, is under Unitree's BSD 3-Clause license (`robots/g1/LICENSE`).

The object and table meshes that `scripts/download_assets.sh` fetches into `data/objects` were generated
with SAM 3D Objects and are distributed under the SAM License. If you publish research that uses them,
the SAM License requires you to acknowledge the use of SAM Materials.

The submodules keep their own licenses. See `THIRD_PARTY_NOTICES` for details.

## Citation

If you use this code or build on LEGS, please cite:

```bibtex
@inproceedings{kim2026legs,
  title     = {LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World},
  author    = {Kim, Hojune and Chen, Timothy and Sun, Jiankai and Osterberg, Lars W. and Chen, Qianzhong and Wang, Ke and Schwager, Mac},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
