# object_gen — iPhone photo → simulator-ready 3D mesh

Pipeline that takes a single iPhone photo of an object and produces a `.glb`
mesh that drops into `data/objects/` (next to its `.stl` collision mesh) for
`scene_config.py` and the renderers.

## Pipeline

```
iphone.jpg + click(x,y)
        │
        ▼  sam_mask.py            (Meta SAM v2 image predictor)
   mask.png + mask_overlay.png
        │
        ▼  sam3d_mesh.py          (Meta SAM 3D Objects — submodules/sam-3d-objects)
   mesh.glb + splat.ply
        │
        ▼  color_refine_ccm.py    (per-channel scale fit under render lighting;
        │                          color_refine.py = Reinhard, optional)
   mesh_refined_ccm.glb + ccm.json
        │
        ▼  render_test.py         (renderer.object_renderer.SimpleObjectRenderer,
                                   d435 fov 0.74, 640x480 — same as sim)
   test_view.png + test_view_refined.png
```

Run end-to-end via `run_pipeline.py`, or invoke each step on its own to iterate
on a single stage (e.g. re-mask without re-running SAM3D).

| step | script | env |
|---|---|---|
| mask | `sam_mask.py` | `sam3d-objects` (+ SAM 2) |
| mesh | `sam3d_mesh.py` | `sam3d-objects` |
| refine | `color_refine_ccm.py` / `color_refine.py` | `legs` or `sam3d-objects` |
| render | `render_test.py` | `legs` or `sam3d-objects` |

`refine` and `render` need only numpy, scipy, Pillow, OpenCV, trimesh and
pyrender (EGL): `legs.yml` plus its pyrender post-create step (SETUP.md,
section 2). `color_refine_ccm.py --apply-ccm` does not render, so it also runs
without a GPU.

## Rebuild the shipped objects

`data/objects/sources/<name>/` holds the inputs of the shipped `orange` and
`plate`: `iphone.jpg`, `click.json`, `mask.png`, `ccm.json`, `config.json`
(real height in meters) and `mesh_raw.glb`, the SAM 3D mesh before color
refinement. `mesh_raw.glb` is too large for git and is downloaded:

```bash
bash scripts/download_assets.sh sources
conda activate legs

# orange: refit the color scale and bake it
python scripts/main/object_gen/run_pipeline.py \
    --source data/objects/sources/orange --steps refine,render

# plate: bake the stored scale (see below)
python scripts/main/object_gen/run_pipeline.py \
    --source data/objects/sources/plate --steps refine,render \
    --apply-ccm data/objects/sources/plate/ccm.json
```

Outputs go to `data/objects/sam3d_runs/<name>/`; the sources are never
overwritten. Both `mesh_refined_ccm.glb` files are byte-identical to
`data/objects/orange.glb` and `data/objects/plate.glb`.

The plate's `ccm.json` is frozen. It was fit from a near top-down render of
the plate, while `color_refine_ccm.py` renders a side view that sees the plate
edge-on; a refit there gives diag(2.53, 2.57, 2.46) instead of the stored
(1.39, 1.47, 1.48), a plate about 40 RGB levels brighter. The orange's refit
reproduces its stored `ccm.json` exactly.

To regenerate the masks and meshes as well, run all four steps from the
`sam3d-objects` env (`--steps mask,mesh,refine,render`); `click.json` and
`config.json` supply the clicks and the real height. SAM 3D is generative, so
the new mesh will not match `mesh_raw.glb` exactly.

The orange and plate `.stl` collision meshes in `data/objects/` (5,000 faces
each) are not produced by this pipeline; [Usage](#end-to-end) shows how to make
one for a new object.

## Setup

### 1. SAM 3D Objects env

Follow `submodules/sam-3d-objects/doc/setup.md` (creates `sam3d-objects` conda
env, installs `.[dev]` + `.[p3d]` + `.[inference]`, runs `./patching/hydra`).
This needs an NVIDIA GPU with at least 32 GB of VRAM, and the pytorch3d build
compiles CUDA code.

### 2. SAM v2 (2D mask) — install into the same env

```bash
mamba activate sam3d-objects
pip install 'git+https://github.com/facebookresearch/sam2.git'
```

The default checkpoint (`facebook/sam2.1-hiera-large`) is auto-downloaded by
`SAM2ImagePredictor.from_pretrained` on first use.

### 3. SAM 3D Objects checkpoints

```bash
pip install 'huggingface-hub[cli]<1.0'   # provides the hf CLI
hf auth login                            # request access first at facebook/sam-3d-objects
cd submodules/sam-3d-objects
TAG=hf
hf download --repo-type model --local-dir checkpoints/${TAG}-download \
  --max-workers 1 facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

## Usage

### End-to-end

From the repository root:

```bash
mamba activate sam3d-objects

# pick the prompt points; prints the matching --click x,y [--neg-click x,y] arguments
python scripts/main/object_gen/pick_clicks.py --image mug.jpg

python scripts/main/object_gen/run_pipeline.py \
    --image mug.jpg \
    --click 612,840 \
    --real-height 0.095 \
    --name mug
```

Output (under `data/objects/sam3d_runs/<name>/`):

```
mug/
├── iphone.jpg              # input copy
├── click.json              # {positive, negative, image_size}
├── mask.png                # binary 0/255
├── mask_overlay.png        # mask drawn on iPhone image (sanity check)
├── mesh.glb                # raw SAM3D mesh
├── splat.ply               # raw SAM3D gaussians
├── mesh_refined_ccm.glb    # color-refined (CCM, default)
├── ccm.json                # fitted per-channel scale + stats
├── ccm_debug_render.png    # unrefined mesh under render lighting
├── test_view.png           # raw mesh @ d435 canonical view
└── test_view_refined.png   # refined mesh @ d435 canonical view
```

With `--refine-method reinhard` the refined mesh is `mesh_refined.glb` instead.
When happy with the result, copy the final asset into the simulator's tree and
make its collision mesh (Z-up, in meters; run in the `legs` env):

```bash
cp data/objects/sam3d_runs/mug/mesh_refined_ccm.glb data/objects/mug.glb
pip install fast_simplification==0.1.13   # trimesh's mesh decimation backend
python -c "import numpy as np, trimesh; m = trimesh.load('data/objects/mug.glb', force='mesh'); m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])); m.simplify_quadric_decimation(face_count=5000).export('data/objects/mug.stl')"
```

Then add an entry with `"mesh": "mug.stl"` to `scripts/main/scene_config.py`
like the other objects. The renderer draws `mug.glb` and MuJoCo simulates
`mug.stl`; the first scene build splits it into convex parts with CoACD
(`data/objects/mug/mug_col_*.stl`).

### Per-step

`run_pipeline.py --steps mask` (or `mesh`, `refine`, `render`) runs only one
stage. Useful when you want to retry the mask click without re-paying the
~minute-long SAM3D inference.

```bash
# fix a bad mask, then re-do everything downstream
python scripts/main/object_gen/run_pipeline.py --image mug.jpg --click 700,820 --name mug --steps mask
python scripts/main/object_gen/run_pipeline.py --name mug --steps mesh,refine,render --real-height 0.095
```

## Knobs

- `--fov` (default 0.74) — vertical FOV in radians, matches D435 in sim.
- `--cam-distance` (default 0.4 m) — canonical camera placed along +X looking
  at origin; mesh is centered to its AABB.
- `--real-height M` — the object's real height in meters. SAM 3D's mesh comes
  out normalized to about a unit cube; it is rescaled so its height (GLB Y
  extent) is M. Pass it for every new object, also when re-running the `mesh`
  step; `--source` runs take it from `config.json`.
- `--no-refine` — skip color refinement if the raw mesh colors look fine.
- `--refine-method reinhard` — Reinhard mean/std transfer in Lab instead of CCM.
- `--apply-ccm CCM_JSON` — bake a stored `ccm.json` instead of refitting.

## Color refinement

SAM3D's vertex colors come out muted and off-tone vs the actual iPhone
colors (the model regresses albedo, not the iPhone ISP's tone curve). The
default, `color_refine_ccm.py`, renders the mesh under the same lighting as
`offline_renderer.py` and fits a per-channel scale in linear RGB so the
rendered object's mean color matches the masked iPhone pixels, then bakes the
scale into the vertex colors. `color_refine.py` instead matches the vertex
colors' Lab mean/std to the iPhone region directly (Reinhard et al. 2001),
which ignores the renderer's shading. Geometry is untouched either way.

Refined meshes stay in the iPhone's color domain, like the gsplat scene;
`offline_renderer.py` maps every composited frame to the RealSense with the
chart-fitted matrix in `renderer/iphone_to_realsense.py` (see
`scripts/calibration/`).
