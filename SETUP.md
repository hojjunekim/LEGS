# Setup

This guide installs the LEGS simulator, its whole-body-control backend, and the environments for
rendering, dataset export, training and splat capture. Run the commands from the repository root
unless a step says otherwise.

| Environment | Created by | Used for |
|---|---|---|
| `legs` (conda) | `legs.yml` | simulator, offline rendering, color calibration, evaluation scripts |
| `.venv_sim` (uv, in the GR00T-WBC submodule) | `install_mujoco_sim.sh` | MuJoCo sim loop of the backend |
| `g1-deploy-dev` (docker) | `run-ros2-dev.sh` | SONIC controller and ROS 2 deploy |
| `submodules/Psi0/.venv-openpi` (uv) | `scripts/train/setup_envs.sh --pi05` | dataset export and π0.5 |
| `.venv-psi`, `src/gr00t/.venv` (uv, in Psi0) | `scripts/train/setup_envs.sh` | ψ0 and GR00T N1.6 |
| `legs-gsplat` (conda) | `gsplat.yml` | capturing a new splat (optional) |

The simulator needs:

- Ubuntu 22.04 with an NVIDIA GPU from Volta to Hopper (compute capability 7.0 to 9.0) and driver 535
  or newer;
- `git`, Git LFS (for the GR00T-WholeBodyControl submodule), `curl` and a C/C++ compiler
  (`sudo apt install git git-lfs curl build-essential`); the `legs` env compiles `evdev` and gsplat's
  CUDA kernels with the system compiler;
- conda, and docker with the NVIDIA Container Toolkit, usable without `sudo` (add your user to the
  `docker` group);
- ROS 2 Humble installed system-wide (`sudo apt install ros-humble-ros-base`, after adding the ROS 2
  apt repository as in the
  [ROS 2 Humble install guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)),
  used only by the backend's sim loop;
- TensorRT 10.13 for the SONIC controller (section 4(c)).

## 1. Clone

Run `git lfs install` once, then clone with submodules:

```bash
git -c url."https://github.com/".insteadOf=git@github.com: clone --recursive https://github.com/hojjunekim/LEGS.git
cd LEGS
```

The submodules are GR00T-WholeBodyControl (backend), Psi0 (training and policy servers) and
sam-3d-objects (object meshes). Psi0 brings `third_party/SIMPLE`, which its uv environments need on disk,
and SIMPLE brings submodules of its own. Some of these are listed with SSH URLs (`git@github.com:…`); the
`-c url…insteadOf` option fetches them over HTTPS, so no GitHub SSH key is needed. The full clone is about
5 GB, 2.4 GB of it GR00T-WholeBodyControl with its LFS files.

If you cloned without `--recursive`, run
`git -c url."https://github.com/".insteadOf=git@github.com: submodule update --init --recursive`. If you
cloned without Git LFS, run `git lfs pull` in `submodules/GR00T-WholeBodyControl`.

`submodules/GR00T-WholeBodyControl` is a fork of `NVlabs/GR00T-WholeBodyControl` on its `sim-handoff`
branch. It adds the following to upstream:

- object pose publishers;
- the scene-XML hookup;
- UDP-only Fast DDS;
- a docker CUDA version match;
- the SONIC LeRobot exporter used by the dataset build.

`submodules/Psi0` is a fork of `physical-superintelligence-lab/Psi0` on its `legs` branch. It adds the
LEGS fine-tuning configs for ψ0, π0.5 and GR00T N1.6, and policy servers for the 18-D state and action.

## 2. Simulator environment (`legs`)

```bash
conda env create -f legs.yml -n legs
conda activate legs
# post-create steps (also listed at the bottom of legs.yml)
pip install --no-deps pyrender==0.1.45
pip install --no-deps -e ./submodules/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python
```

`source ./init.sh` activates the env and sets the DDS transport. ROS 2 Humble comes from RoboStack
inside the env, so do not also source the system ROS install in a `legs` shell. Only the backend's sim
loop (Shell A in section 5) uses the system install.

## 3. Assets

```bash
bash scripts/download_assets.sh core      # G1, object and table meshes, flightroom splat (186.5 MB archive)
bash scripts/download_assets.sh paper     # wood-table splat + HDR-refit orange and plate (128.0 MB)
```

The simulator, the scene build and the renderers need `core`. The renderer needs `paper` for the
wood-table background of the paper's datasets. `aug` (appearance study) and `sources` (mesh-only baseline
and object rebuilds) are optional. The README's "Large assets" section lists all groups.

## 4. Backend: GR00T-WholeBodyControl

The backend splits into a host-side MuJoCo sim loop and a docker-side SONIC controller with the ROS 2
deploy.

**(a) Host venv** (one-time; creates `.venv_sim/` in the submodule):

```bash
cd submodules/GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh    # installs uv, Python 3.10, gear_sonic[sim]
export PATH="$HOME/.local/bin:$PATH"          # where uv's installer puts uv, if the script installed it
uv pip install --python .venv_sim/bin/python mujoco==3.5.0
```

`gear_sonic[sim]` does not pin MuJoCo, so the installer takes the newest release. The last line pins
the version the paper's episodes were recorded with. Rerunning the installer recreates `.venv_sim`, so
run the `uv pip install` line again after it.

**(b) SONIC checkpoints** (`nvidia/GEAR-SONIC` on Hugging Face; written to
`gear_sonic_deploy/policy/release/` and `gear_sonic_deploy/planner/`):

```bash
conda activate legs                            # any env with huggingface_hub
python download_from_hf.py                     # from submodules/GR00T-WholeBodyControl
```

**(c) Docker image and controller build** (one-time). The SONIC controller runs on TensorRT, which
the launcher mounts from the host. Download the TensorRT 10.13 TAR package for x86_64 and CUDA 12 from
[NVIDIA](https://developer.nvidia.com/tensorrt/download/10x), extract it (for example to `~/TensorRT`),
and set `TensorRT_ROOT` (add the export to `~/.bashrc` so later shells have it):

```bash
export TensorRT_ROOT=$HOME/TensorRT            # the extracted TensorRT directory
cd gear_sonic_deploy/docker
./run-ros2-dev.sh                              # first run builds the image (g1-deploy-dev), then opens a shell in it
# inside the container:
source scripts/setup_env.sh
just build                                     # compiles the controller into target/release/
exit
cd ../../../..                                 # back to the repository root
```

`deploy.sh` does not build, so rerun `just build` in the container after changing the C++ sources. The
container runs as root, so what it writes into `gear_sonic_deploy/` is owned by root: `build/`,
`target/`, the TensorRT engines (`*.trt`, section 5) and `collect_logs/` (section 6). Delete them from
inside the container or with `sudo`.

If one of these steps fails, see upstream's
[installation guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html).

## 5. Run the simulator

First generate the MuJoCo scene XML for your scene. `scene_config.py` writes it to
`$GROOT_WBC_DIR/gear_sonic/data/robot_model/model_data/g1/generated_scene.xml`, and the sim loop
(Shell A) loads it at startup, so rerun it after each scene change and restart the backend.
Collision comes from the convex parts in `data/objects/<name>/`. The `core` download includes the
paper's parts for the tables, orange and plate, and any other mesh is decomposed with CoACD on its first
build:

```bash
source ./init.sh
(cd scripts/main && python scene_config.py orange_to_plate_locomani)
```

Then start three shells, in this order.

**Shell A**, the MuJoCo sim loop (host), in a new shell without the `legs` env:

```bash
source /opt/ros/humble/setup.bash             # the sim loop takes rclpy from the system ROS 2 Humble
cd submodules/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

Over ROS 2 it publishes the object poses (`/simulation/pose/<name>`) and serves `/simulation/reset`,
which the frontend uses.

**Shell B**, the SONIC controller (inside docker), once the MuJoCo window is open:

```bash
submodules/GR00T-WholeBodyControl/gear_sonic_deploy/docker/run-ros2-dev.sh
# inside the container:
cd /workspace/g1_deploy
bash deploy.sh sim --input-type manager
```

`deploy.sh` prints the command it will run and asks `Proceed with deployment? [Y/n]`; press Enter. On
the first run it then builds TensorRT engines for its three models (about 2.5 minutes on an RTX 3090)
and saves them for later runs. The controller then needs robot state from the sim loop within 5
seconds. Without it, it stops with `Planner initialization timeout after 5 seconds`. When it prints
`Planner initialized successfully!`, click the MuJoCo window and press `9` to release the virtual
elastic band that holds the robot up. The band stays released when the frontend resets the sim.

Each `run-ros2-dev.sh` call first removes a running `g1-deploy-dev` container, controller included. For a
second shell in the container, use `docker exec -it g1-deploy-dev bash`, then
`source scripts/setup_env.sh`.

**Shell C**, the LEGS frontend (host):

```bash
source ./init.sh
cd scripts/main
python simulate_object.py --scene orange_to_plate_locomani
```

Open `http://localhost:8080` for the viser UI. On the first start, gsplat compiles its CUDA kernels at
the first render, which takes about a minute and a half; the viser page can look stuck until it
finishes. The build uses `CUDA_HOME` if it is set, otherwise the `legs` env's CUDA 11.8 toolkit. It
compiles four files in parallel, which can take more than 8 GB of RAM; if it is killed for lack of
memory, start the frontend with `MAX_JOBS=2`. CUDA 11.8 cannot build for Blackwell GPUs (RTX 50xx,
B200); on those, set `CUDA_HOME` to a CUDA 12.8 or newer toolkit before the first start.

The frontend and the backend talk over ROS 2 / DDS on domain 0, so run one stack per machine.

## 6. End-to-end collection

Once the simulator works, `run_collection.sh` chains the steps:

1. regenerate the scene XML;
2. record episodes;
3. snapshot the deploy CSVs;
4. render;
5. build the LeRobot dataset.

```bash
./run_collection.sh <N_EPISODES> <NAME> <SCENE>    # e.g. ./run_collection.sh 200 my_turn orange_to_plate_turn
```

With the default `STOP_AFTER=7` the script also renders and builds the dataset, so before recording it
checks for the dataset-export env and stops if `submodules/Psi0/.venv-openpi` is missing. Build that env
first (`bash scripts/train/setup_envs.sh --pi05`, section 7), or record only with
`STOP_AFTER=5 ./run_collection.sh <N_EPISODES> <NAME> <SCENE>`.

Before running it, open the container in Shell B (`run-ros2-dev.sh`); the script checks that it is
running. After regenerating the scene XML, it tells you to start the sim loop and then the deploy, with
`EXTRA_DEPLOY_ARGS="--enable-csv-logs --logs-dir collect_logs"` so the SONIC motion tokens are logged.
Release the elastic band as in section 5 and press Enter. The script then launches the frontend; click
**🔴 Auto Record** in viser to begin.

Recordings go to `data/collected/<NAME>_<timestamp>/` and the dataset to
`data/lerobot/wigs/<NAME>_uni/`. The header of `run_collection.sh` documents `STOP_AFTER` and `PLY`.

| Variable | Default | What it points at |
|---|---|---|
| `GROOT_WBC_DIR` | `submodules/GR00T-WholeBodyControl` | backend checkout; the scene XML is written into it |
| `GROOT_WBC_GENERATED_XML` | `$GROOT_WBC_DIR/gear_sonic/data/robot_model/model_data/g1/generated_scene.xml` | full path to the generated scene XML |
| `CONDA_BASE` | `conda info --base` | conda install root (used by `run_collection.sh`) |
| `LEGS_DATA_PY` | `submodules/Psi0/.venv-openpi/bin/python` | Python that builds datasets |

## 7. Dataset export

### Environment

`scripts/data/*2universal.sh` convert recorded episodes into LeRobot v2.1 datasets. They run in Psi0's
`.venv-openpi` (the exporter needs lerobot < 0.3), with `gear_sonic` installed from this repo's
GR00T-WholeBodyControl submodule:

```bash
bash scripts/train/setup_envs.sh --pi05        # builds submodules/Psi0/.venv-openpi, installs gear_sonic
# must print a path under submodules/GR00T-WholeBodyControl:
env -u PYTHONPATH submodules/Psi0/.venv-openpi/bin/python -c "import gear_sonic.data.exporter as e; print(e.__file__)"
```

The scripts use `$LEGS_DATA_PY` if set, otherwise `submodules/Psi0/.venv-openpi/bin/python`, and clear
`PYTHONPATH` (the `legs` env's ROS 2 sets it, and `conda deactivate` leaves it set). If the check prints
a different checkout, re-run `bash scripts/train/setup_envs.sh --pi05`, which reinstalls `gear_sonic`
from this repo.

### Building a dataset

```bash
# gsplat renders (final_renders/)
bash scripts/data/wigs2universal.sh <raw-dir> <out-dir> [<raw-dir>/csv_snapshot | ""] [task]
# mesh-only baseline renders (baseline_renders/); same arguments
bash scripts/data/sam2universal.sh <raw-dir> <out-dir> [<raw-dir>/csv_snapshot | ""] [task]
# several runs or episode ranges in one dataset; extra flags go to build_sonic_lerobot.py
bash scripts/data/build_universal_multi.sh <wigs|sam> <out-dir> \
    --source-data-dir <raw-dir> --source-ep-from N --source-ep-to M --source-task "<prompt>" [...]
# teleop recordings: converts an existing LeRobot dataset in place
bash scripts/data/teleop2universal.sh <dataset-root>
```

- `<out-dir>` is overwritten.
- The third argument is searched recursively for `token_state.csv` (SONIC motion tokens).
  - A path that doesn't exist is an error.
  - `""` zero-fills `action.motion_token`. That is fine for ψ0, π0.5 and GR00T N1.6, but not for SONIC.
- `[task]` overrides the recorded prompt. Use it for appearance variants such as `apple_to_box_*`.
- Every episode is checked before anything is written. A missing scenario file, render directory or frame
  stops the build. To override, pass builder flags after `--`, for example
  `bash scripts/data/wigs2universal.sh <raw-dir> <out-dir> "" -- --allow-skip --allow-missing-frames`.

Building takes about 7 ms per frame on CPU, which is roughly 15–30 minutes for a 200-episode dataset.

The output is LeRobot v2.1 at 30 fps. It contains:

- video `observation.images.ego_view` (480×640, H.264);
- ψ0 state `observation.state_psi0_18` (18-D; the older 15-D `observation.state_psi0` is also kept) and
  action `action.psi0_18`;
- universal `observation.state_uni_20` / `action.uni_23`, and G1 loco `observation.state_g1_loco_32` /
  `action.g1_loco_36`;
- `action.motion_token`;
- `meta/stats.json` (ψ0), `norm_stats.json` (π0.5) and `meta/modality*.json` sidecars for ψ0 and GR00T.

## 8. Training environments

Training needs [uv](https://docs.astral.sh/uv/) and NVIDIA GPUs with a CUDA 12 driver. Run
`bash scripts/train/setup_envs.sh` for all three environments, or pass `--psi0`, `--pi05` or `--groot`.
If no prebuilt flash-attn wheel matches your platform, it is built from source. That needs `nvcc` from a
CUDA 12 toolkit, with `CUDA_HOME` set. See [`scripts/train/README.md`](scripts/train/README.md) for
datasets, weights and launch commands.

The policy servers for evaluation run in the same environments (see
[`scripts/eval/README.md`](scripts/eval/README.md)). On the simulator side, the π0.5 client needs
`websockets` and `openpi-client`; the `legs` env has `websockets`, and `openpi-client` installs from the
Psi0 submodule with `pip install -e submodules/Psi0/src/openpi/openpi-client`.

## 9. Object generation (optional)

Color refinement and test renders run in the `legs` env. Making new masks or meshes needs the
`sam3d-objects` env from `submodules/sam-3d-objects/doc/setup.md`, plus
`pip install 'git+https://github.com/facebookresearch/sam2.git'` and the gated `facebook/sam-3d-objects`
checkpoints. The steps are in [`scripts/main/object_gen/README.md`](scripts/main/object_gen/README.md).

## 10. Splat capture (optional)

```bash
conda env create -f gsplat.yml               # creates legs-gsplat
conda activate legs-gsplat
python scripts/gsplat/video2gsplat.py <name> --video <capture.mov>
```

gsplat compiles its CUDA kernels against the env's CUDA 11.8 toolkit on first use, which takes a few
minutes. The README's "Capturing a new splat" section covers calibration.
