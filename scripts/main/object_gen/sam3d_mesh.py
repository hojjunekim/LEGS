"""Step 2: iPhone JPG + 2D mask -> 3D mesh.glb + splat.ply via Meta SAM 3D Objects.

Wraps `submodules/sam-3d-objects/notebook/inference.py` Inference class.
Output dict from Inference(...) has both `output["glb"]` (trimesh.Scene with
baked vertex colors, exportable) and `output["gs"]` (Gaussian splat with
`.save_ply(...)`).

Outputs (under <out_dir>):
    mesh.glb     # vertex-colored mesh from SAM3D (use_vertex_color=True)
    splat.ply    # gaussian splat for optional viewer use

Usage:
    python sam3d_mesh.py \
        --image data/objects/sam3d_runs/mug/iphone.jpg \
        --mask  data/objects/sam3d_runs/mug/mask.png \
        --out   data/objects/sam3d_runs/mug \
        --sam3d-root submodules/sam-3d-objects
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def _load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _load_mask(path: str) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., -1]
    return (m > 0).astype(bool)


def _scale_glb_to_height(glb_path: Path, real_height_m: float) -> float:
    """Rescale a saved GLB so its Y-extent (GLB up convention) == real_height_m.

    SAM3D outputs are normalized to a ~unit cube; the sim needs metric units.
    Returns the scale factor applied.
    """
    scene = trimesh.load(str(glb_path))
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    y_extent = float(mesh.extents[1])
    if y_extent <= 0:
        raise ValueError(f"{glb_path}: Y-extent is 0 — cannot scale to {real_height_m}m")
    scale = real_height_m / y_extent
    scene.apply_transform(np.diag([scale, scale, scale, 1.0]).astype(np.float32))
    scene.export(str(glb_path))
    return scale


def run(image_path: str, mask_path: str, out_dir: str,
        sam3d_root: str, checkpoint_tag: str = "hf",
        seed: int = 42, real_height_m: float | None = None) -> dict:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Resolve I/O paths to absolute — we chdir into sam3d_root below, which
    # would break any relative paths the caller passed in.
    image_path = str(Path(image_path).resolve())
    mask_path = str(Path(mask_path).resolve())

    sam3d_root = str(Path(sam3d_root).resolve())
    nb_dir = os.path.join(sam3d_root, "notebook")
    if nb_dir not in sys.path:
        sys.path.insert(0, nb_dir)

    # Inference reads `checkpoints/<tag>/pipeline.yaml` relative to sam3d_root,
    # so we chdir into it for the model load (matches demo.py expectations).
    config_rel = f"checkpoints/{checkpoint_tag}/pipeline.yaml"
    config_abs = os.path.join(sam3d_root, config_rel)
    if not os.path.exists(config_abs):
        raise FileNotFoundError(
            f"SAM3D checkpoint config not found: {config_abs}\n"
            f"  Run the HF download steps from {sam3d_root}/doc/setup.md"
        )

    prev_cwd = os.getcwd()
    os.chdir(sam3d_root)
    try:
        from inference import Inference  # type: ignore
        print(f"  Loading SAM3D pipeline from {config_rel} ...")
        inference = Inference(config_rel, compile=False)

        image = _load_image(image_path)
        mask = _load_mask(mask_path)
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"mask {mask.shape} != image {image.shape[:2]} — re-run sam_mask.py"
            )

        print(f"  Running SAM3D inference (seed={seed})...")
        output = inference(image, mask, seed=seed)
    finally:
        os.chdir(prev_cwd)

    mesh_path = out / "mesh.glb"
    splat_path = out / "splat.ply"

    glb = output.get("glb")
    if glb is None:
        raise RuntimeError("SAM3D returned no 'glb' output — check pipeline config")
    glb.export(str(mesh_path))
    print(f"  wrote {mesh_path}")

    if real_height_m is not None:
        s = _scale_glb_to_height(mesh_path, real_height_m)
        print(f"  scaled mesh.glb to {real_height_m}m Y-extent (scale={s:.4f})")

    gs = output.get("gs")
    if gs is not None:
        gs.save_ply(str(splat_path))
        print(f"  wrote {splat_path} (NOTE: splat NOT scaled — viewer-only)")
    else:
        print("  WARNING: no 'gs' output, skipping splat.ply")

    return {"mesh": str(mesh_path), "splat": str(splat_path) if gs is not None else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--sam3d-root", default=None,
        help="path to submodules/sam-3d-objects (default: auto-detect from repo)",
    )
    p.add_argument("--checkpoint-tag", default="hf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real-height", type=float, default=None,
                   help="if set, scale mesh.glb so its Y-extent (GLB up) equals this many meters")
    args = p.parse_args()

    if args.sam3d_root is None:
        here = Path(__file__).resolve()
        # scripts/main/object_gen/ -> repo root -> submodules/sam-3d-objects
        repo_root = here.parents[3]
        args.sam3d_root = str(repo_root / "submodules" / "sam-3d-objects")

    run(args.image, args.mask, args.out, args.sam3d_root,
        checkpoint_tag=args.checkpoint_tag, seed=args.seed,
        real_height_m=args.real_height)


if __name__ == "__main__":
    main()
