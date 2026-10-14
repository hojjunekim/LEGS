"""Step 3: Reinhard color transfer from masked iPhone region -> mesh vertex colors.

SAM3D meshes come out with vertex colors that approximate the input texture
but with shifted tone/saturation (model regresses albedo without the iPhone
ISP's tone curve). We pull the *target* color distribution from the masked
iPhone pixels and shift the mesh's vertex colors to match its mean+std
per-channel in CIE Lab — this is the classic Reinhard et al. (2001) transfer.

Geometry is untouched. Only `mesh.visual.vertex_colors` (or the PBR texture's
baseColorTexture, downgraded to vertex colors if present) is rewritten.

Not the default: run_pipeline.py refines with color_refine_ccm.py, which matches
the mesh under the renderer's lighting, and uses this only with
`--refine-method reinhard`. The shipped orange and plate are CCM outputs.

Outputs (under <out_dir>):
    mesh_refined.glb

Usage:
    python color_refine.py \
        --image  data/objects/sam3d_runs/mug/iphone.jpg \
        --mask   data/objects/sam3d_runs/mug/mask.png \
        --mesh   data/objects/sam3d_runs/mug/mesh.glb \
        --out    data/objects/sam3d_runs/mug
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image


def _rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB (..., 3) -> float32 Lab via OpenCV."""
    shape = rgb_u8.shape
    flat = rgb_u8.reshape(-1, 1, 3).astype(np.uint8)
    bgr = cv2.cvtColor(flat, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab.reshape(shape)


def _lab_to_rgb(lab_f32: np.ndarray) -> np.ndarray:
    shape = lab_f32.shape
    flat = np.clip(lab_f32, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(flat, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.reshape(shape)


def reinhard_transfer(src_rgb_u8: np.ndarray, tgt_rgb_u8: np.ndarray) -> np.ndarray:
    """Shift `src` colors to match `tgt` per-channel mean/std in Lab.

    Both inputs are (N, 3) uint8 (sample sets, not images). Returns (N_src, 3)
    uint8 with `src.shape[0]` rows.
    """
    src_lab = _rgb_to_lab(src_rgb_u8.reshape(-1, 3))
    tgt_lab = _rgb_to_lab(tgt_rgb_u8.reshape(-1, 3))

    s_mean = src_lab.mean(axis=0)
    s_std = src_lab.std(axis=0) + 1e-6
    t_mean = tgt_lab.mean(axis=0)
    t_std = tgt_lab.std(axis=0) + 1e-6

    out_lab = (src_lab - s_mean) * (t_std / s_std) + t_mean
    return _lab_to_rgb(out_lab)


def _gather_mesh_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Return (N,3) uint8 RGB, or None if mesh has no usable color info.

    Handles both pure vertex-color meshes and textured meshes (samples the
    texture at vertex UVs).
    """
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None

    vc = getattr(visual, "vertex_colors", None)
    if vc is not None and len(vc) == len(mesh.vertices):
        return np.asarray(vc, dtype=np.uint8)[:, :3]

    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    tex_img = None
    if material is not None:
        tex_img = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if uv is not None and tex_img is not None:
        tex = np.array(tex_img.convert("RGB"))
        H, W = tex.shape[:2]
        u = np.clip(uv[:, 0], 0, 1) * (W - 1)
        v = (1.0 - np.clip(uv[:, 1], 0, 1)) * (H - 1)  # flip V (image origin top)
        return tex[v.astype(np.int32), u.astype(np.int32)]

    return None


def _srgb_to_linear_u8(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> linear uint8 using the standard sRGB EOTF.

    glTF 2.0 says COLOR_0 vertex colors are linear, and pyrender's mesh.frag
    gamma-encodes (pow 1/2.2) on output. Reinhard runs in sRGB-derived Lab,
    so its output is sRGB; we must linearize before storing or pyrender will
    double-encode and brighten the render.
    """
    s = rgb_u8.astype(np.float32) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    return np.clip(lin * 255.0, 0, 255).astype(np.uint8)


def _set_mesh_vertex_colors(mesh: trimesh.Trimesh, rgb_u8: np.ndarray) -> trimesh.Trimesh:
    """Strip texture (if any) and set per-vertex RGBA colors (linear, glTF-spec)."""
    lin_u8 = _srgb_to_linear_u8(rgb_u8)
    rgba = np.concatenate(
        [lin_u8, 255 * np.ones((len(lin_u8), 1), dtype=np.uint8)], axis=1
    )
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    return mesh


def run(image_path: str, mask_path: str, mesh_path: str, out_dir: str) -> dict:
    if not Path(mesh_path).exists():
        hint = ("; fetch the source meshes with `bash scripts/download_assets.sh sources`"
                if Path(mesh_path).name == "mesh_raw.glb" else "")
        raise FileNotFoundError(f"missing mesh {mesh_path}{hint}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., -1]
    mask_bool = mask > 0
    target_pixels = image[mask_bool]
    if target_pixels.shape[0] < 100:
        raise ValueError(
            f"masked region has only {target_pixels.shape[0]} pixels — too small"
        )

    scene_or_mesh = trimesh.load(mesh_path)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.dump(concatenate=True)
    else:
        mesh = scene_or_mesh
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"loaded {mesh_path} as {type(mesh).__name__}, expected Trimesh")

    src_rgb = _gather_mesh_vertex_colors(mesh)
    if src_rgb is None:
        raise RuntimeError(
            f"{mesh_path} has neither vertex colors nor a texture — nothing to refine"
        )

    print(f"  src: {len(src_rgb)} vertex colors")
    print(f"  tgt: {target_pixels.shape[0]} masked iPhone pixels")
    new_rgb = reinhard_transfer(src_rgb, target_pixels)
    mesh = _set_mesh_vertex_colors(mesh, new_rgb)

    out_path = out / "mesh_refined.glb"
    mesh.export(str(out_path))
    print(f"  wrote {out_path}")

    src_mean = src_rgb.mean(axis=0)
    tgt_mean = target_pixels.mean(axis=0)
    new_mean = new_rgb.mean(axis=0)
    print(f"  RGB means: src={src_mean.round(1).tolist()} "
          f"tgt={tgt_mean.round(1).tolist()} out={new_mean.round(1).tolist()}")
    return {"mesh": str(out_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run(args.image, args.mask, args.mesh, args.out)


if __name__ == "__main__":
    main()
