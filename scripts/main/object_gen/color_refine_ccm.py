"""Step 3 (alt): CCM-based color calibration via post-render mean match.

Difference vs. `color_refine.py` (Reinhard):
  - Reinhard shifts vertex colors so the *vertex* distribution matches the
    iPhone pixel distribution. But pyrender applies PBR lighting (ambient +
    headlight) at render time, so the rendered image is muted/shifted
    relative to the iPhone photo.
  - CCM (here): render the mesh under DEPLOYMENT lighting (intensity=1.0,
    ambient=0.3, headlight — the SimpleObjectRenderer defaults that
    `offline_renderer.py` uses), then fit a per-channel diagonal scale
    in *linear* RGB so the rendered pixel mean matches the iPhone pixel mean.
    Bake the scale into vertex colors.

Why diagonal (not a full 3x3 like the iPhone→RealSense CCM in
renderer/iphone_to_realsense.py):
  - With pyrender deployment lighting the rendered orange is ~2x dimmer than
    the iPhone photo. A full 3x3 fit to match both mean *and* covariance ends
    up with diagonal entries of 6+ to overcome the std gap (iPhone has wide
    intra-region variation from shading/highlights that the render lacks).
    Multiplying vertex colors by 6 blows past uint8 255 and clips, turning
    the orange green.
  - A diagonal mean-only scale (e.g. [2.14, 1.47, 0.72] for the orange) stays
    far below that, and the rendered mean lands within ~10 RGB units of the
    iPhone target. Bright channels still saturate (R = 255 on 99% of the
    orange's vertices, 28-37% of the plate's per channel). Loses inter-channel
    coupling (no hue rotation) but preserves brightness and rough hue.

With --apply-ccm, a stored ccm.json is baked into the mesh as-is: no render,
no refit, no GL context. The shipped plate uses this (see
data/objects/sources/plate/config.json).

Outputs (under <out_dir>):
    mesh_refined_ccm.glb
    ccm.json    # diag(scale), b=0, source/target/predicted means
    ccm_debug_render.png    # rendered uncalibrated mesh (fit mode only)

Usage:
    python color_refine_ccm.py \\
        --image  data/objects/sources/orange/iphone.jpg \\
        --mask   data/objects/sources/orange/mask.png \\
        --mesh   data/objects/sources/orange/mesh_raw.glb \\
        --out    data/objects/sam3d_runs/orange

    python color_refine_ccm.py \\
        --mesh   data/objects/sources/plate/mesh_raw.glb \\
        --apply-ccm data/objects/sources/plate/ccm.json \\
        --out    data/objects/sam3d_runs/plate
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYRENDER_OFFSCREEN", "1")

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).resolve().parent
_REPO_MAIN = _HERE.parent
if str(_REPO_MAIN) not in sys.path:
    sys.path.insert(0, str(_REPO_MAIN))

_DOWNLOAD_HINT = "fetch the source meshes with `bash scripts/download_assets.sh sources`"


def require_mesh(mesh_path) -> None:
    """Fail early, naming the download step when a source mesh is missing."""
    if not Path(mesh_path).exists():
        hint = f"; {_DOWNLOAD_HINT}" if Path(mesh_path).name == "mesh_raw.glb" else ""
        raise FileNotFoundError(f"missing mesh {mesh_path}{hint}")


# ---------- sRGB <-> linear ----------

def _srgb_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB -> float64 linear RGB in [0, 1]."""
    s = rgb_u8.astype(np.float64) / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb_u8(rgb_lin: np.ndarray) -> np.ndarray:
    """float64 linear RGB in [0, 1] -> uint8 sRGB."""
    s = np.where(
        rgb_lin <= 0.0031308,
        rgb_lin * 12.92,
        1.055 * (np.clip(rgb_lin, 0, None) ** (1.0 / 2.4)) - 0.055,
    )
    return np.clip(s * 255.0, 0, 255).astype(np.uint8)


# ---------- Mesh I/O (mirrors color_refine.py) ----------

def _gather_mesh_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
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
        v = (1.0 - np.clip(uv[:, 1], 0, 1)) * (H - 1)
        return tex[v.astype(np.int32), u.astype(np.int32)]
    return None


def _set_mesh_vertex_colors_linear(mesh: trimesh.Trimesh, rgb_lin_u8: np.ndarray) -> trimesh.Trimesh:
    """Set per-vertex RGBA. Input is already linear uint8 (glTF spec)."""
    rgba = np.concatenate(
        [rgb_lin_u8, 255 * np.ones((len(rgb_lin_u8), 1), dtype=np.uint8)], axis=1
    )
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
    return mesh


def _load_mesh_for_render(path: str) -> trimesh.Trimesh:
    """Match render_test._load_mesh: .glb gets X-90 (Y-up GLB -> Z-up world)."""
    loaded = trimesh.load(path)
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"loaded {path} as {type(mesh).__name__}, expected Trimesh")
    if path.lower().endswith(".glb"):
        correction = np.eye(4, dtype=np.float32)
        correction[:3, :3] = R.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
        mesh.apply_transform(correction)
    return mesh


# ---------- Rendering ----------

def _look_at_opencv(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = eye
    return T


def _render_under_deployment_lighting(
    mesh: trimesh.Trimesh, width: int, height: int,
    fov: float, cam_distance: float,
    intensity_mult: float = 1.0, ambient: float = 0.3,
) -> np.ndarray:
    """Render at the canonical view with offline_renderer-equivalent lighting.

    Returns RGBA uint8 (sRGB-encoded, per pyrender's gamma-encoding pipeline).
    """
    # Imported here so that --apply-ccm runs without pyrender / a GL context.
    from renderer.object_renderer import SimpleObjectRenderer

    aabb_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh = mesh.copy()
    mesh.apply_translation(-aabb_center)

    rndr = SimpleObjectRenderer(viewport_width=width, viewport_height=height)
    rndr.set_mesh(mesh, smooth=False)
    rndr.set_lighting(intensity_mult=intensity_mult, ambient=ambient)

    eye = np.array([cam_distance, 0.0, 0.05], dtype=np.float32)
    target = np.zeros(3, dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    T_cam = _look_at_opencv(eye, target, up)
    T_obj = np.eye(4, dtype=np.float32)
    rgba, _ = rndr.render(T_cam, T_obj, scale=1.0, width=width, height=height, fov_rad=fov)
    return rgba


# ---------- CCM fit ----------

def _fit_ccm_diag_mean(
    src_lin: np.ndarray, tgt_lin: np.ndarray,
    scale_clamp: tuple[float, float] = (0.5, 3.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel diagonal scale so mean(scale * src) == mean(tgt).

    Returns (M, b) with M = diag(scale_clamped) and b = zeros, so the rest of
    the pipeline can treat it as a generic 3x3 + offset. Scales are clamped
    to keep extreme brightness mismatches from blowing vertex colors past
    uint8 range.
    """
    mean_s = src_lin.mean(axis=0)
    mean_t = tgt_lin.mean(axis=0)
    scale = (mean_t + 1e-9) / (mean_s + 1e-9)
    lo, hi = scale_clamp
    scale_clamped = np.clip(scale, lo, hi)
    M = np.diag(scale_clamped)
    b = np.zeros(3)
    if not np.allclose(scale, scale_clamped):
        print(f"  diag scale clamped from {scale.round(3).tolist()} "
              f"to {scale_clamped.round(3).tolist()} (range {lo}..{hi})")
    return M, b


def _apply_ccm(rgb_lin: np.ndarray, M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """y = M @ x + b, broadcast over leading dims. Input/output linear RGB."""
    return rgb_lin @ M.T + b


# ---------- Bake ----------

def _bake_vertex_colors(mesh_path: str, M: np.ndarray, b: np.ndarray, out: Path) -> Path:
    """Apply M, b to the mesh's vertex colors and export mesh_refined_ccm.glb."""
    mesh_orig_loaded = trimesh.load(mesh_path)
    mesh_orig = (
        mesh_orig_loaded.dump(concatenate=True)
        if isinstance(mesh_orig_loaded, trimesh.Scene)
        else mesh_orig_loaded
    )
    vc = _gather_mesh_vertex_colors(mesh_orig)
    if vc is None:
        raise RuntimeError(f"{mesh_path} has no vertex colors or texture to refine")

    # glTF stores vertex colors as linear floats; trimesh exposes them as
    # uint8 0..255 — those values map to linear [0,1] in the rendering pipeline.
    vc_lin = vc.astype(np.float64) / 255.0
    new_vc_lin = _apply_ccm(vc_lin, M, b)
    new_vc_u8 = np.clip(new_vc_lin * 255.0, 0, 255).astype(np.uint8)

    mesh_orig = _set_mesh_vertex_colors_linear(mesh_orig, new_vc_u8)

    out_mesh_path = out / "mesh_refined_ccm.glb"
    mesh_orig.export(str(out_mesh_path))
    print(f"  wrote {out_mesh_path}")
    return out_mesh_path


def apply(mesh_path: str, ccm_path: str, out_dir: str) -> dict:
    """Bake a stored ccm.json (M, b) into the mesh without rendering or refitting."""
    require_mesh(mesh_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = json.loads(Path(ccm_path).read_text())
    M = np.asarray(info["M"], dtype=np.float64)
    b = np.asarray(info.get("b", [0.0, 0.0, 0.0]), dtype=np.float64)
    print(f"  applying M={M.round(4).tolist()} b={b.round(4).tolist()} from {ccm_path}")
    out_mesh_path = _bake_vertex_colors(mesh_path, M, b, out)

    json_path = out / "ccm.json"
    if json_path.resolve() != Path(ccm_path).resolve():
        shutil.copyfile(ccm_path, json_path)
        print(f"  wrote {json_path}")
    return {"mesh": str(out_mesh_path), "ccm_json": str(json_path)}


# ---------- Main run ----------

def run(image_path: str, mask_path: str, mesh_path: str, out_dir: str,
        width: int = 640, height: int = 480,
        fov: float = 0.74, cam_distance: float = 0.4,
        alpha_thresh: int = 200,
        intensity_mult: float = 1.0, ambient: float = 0.3,
        scale_clamp: tuple[float, float] = (0.5, 3.0)) -> dict:
    require_mesh(mesh_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- iPhone target distribution (in linear RGB) ---
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., -1]
    target_pixels = image[mask > 0]
    if target_pixels.shape[0] < 100:
        raise ValueError(f"masked iPhone region has only {target_pixels.shape[0]} pixels")
    target_lin = _srgb_to_linear(target_pixels)

    # --- Render mesh under deployment lighting, collect source distribution ---
    mesh_for_render = _load_mesh_for_render(mesh_path)
    rgba = _render_under_deployment_lighting(
        mesh_for_render, width, height, fov, cam_distance,
        intensity_mult=intensity_mult, ambient=ambient,
    )
    Image.fromarray(rgba).save(out / "ccm_debug_render.png")

    render_rgb = rgba[..., :3]
    render_alpha = rgba[..., 3]
    interior = render_alpha > alpha_thresh
    src_pixels = render_rgb[interior]
    if src_pixels.shape[0] < 100:
        raise ValueError(
            f"render produced only {src_pixels.shape[0]} interior pixels "
            f"(alpha>{alpha_thresh}) — increase mesh size or lower threshold"
        )
    src_lin = _srgb_to_linear(src_pixels)

    print(f"  iPhone target: {target_pixels.shape[0]} pixels")
    print(f"  render source: {src_pixels.shape[0]} pixels (alpha>{alpha_thresh})")

    # --- Fit CCM (linear RGB) — diagonal mean-only ---
    M, b = _fit_ccm_diag_mean(src_lin, target_lin, scale_clamp=scale_clamp)

    # --- Apply M, b to mesh vertex colors (stored linear per glTF) ---
    out_mesh_path = _bake_vertex_colors(mesh_path, M, b, out)

    # --- Save M, b, and stats ---
    pred_lin = _apply_ccm(src_lin, M, b)
    info = {
        "M": M.tolist(),
        "b": b.tolist(),
        "stats_linear": {
            "src_mean": src_lin.mean(axis=0).tolist(),
            "tgt_mean": target_lin.mean(axis=0).tolist(),
            "pred_mean": pred_lin.mean(axis=0).tolist(),
            "src_std": src_lin.std(axis=0).tolist(),
            "tgt_std": target_lin.std(axis=0).tolist(),
            "pred_std": pred_lin.std(axis=0).tolist(),
        },
        "stats_srgb_u8": {
            "src_mean": src_pixels.mean(axis=0).tolist(),
            "tgt_mean": target_pixels.mean(axis=0).tolist(),
            "pred_mean": _linear_to_srgb_u8(
                np.clip(pred_lin, 0, 1)
            ).mean(axis=0).tolist(),
        },
        "render_settings": {
            "width": width, "height": height, "fov": fov,
            "cam_distance": cam_distance,
            "intensity_mult": intensity_mult, "ambient": ambient,
            "alpha_thresh": alpha_thresh,
        },
    }
    json_path = out / "ccm.json"
    json_path.write_text(json.dumps(info, indent=2))
    print(f"  wrote {json_path}")

    sm = info["stats_srgb_u8"]["src_mean"]
    tm = info["stats_srgb_u8"]["tgt_mean"]
    pm = info["stats_srgb_u8"]["pred_mean"]
    print(f"  sRGB means: src={[round(x,1) for x in sm]} "
          f"tgt={[round(x,1) for x in tm]} pred={[round(x,1) for x in pm]}")

    return {"mesh": str(out_mesh_path), "ccm_json": str(json_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="iPhone JPG (fit mode)")
    p.add_argument("--mask", help="object mask PNG (fit mode)")
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--apply-ccm", default=None, metavar="CCM_JSON",
                   help="bake M, b from this ccm.json instead of fitting from a render")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fov", type=float, default=0.74)
    p.add_argument("--cam-distance", type=float, default=0.4)
    p.add_argument("--alpha-thresh", type=int, default=200,
                   help="alpha > this counts as interior pixel for stats")
    p.add_argument("--intensity-mult", type=float, default=1.0,
                   help="directional light intensity (SimpleObjectRenderer default)")
    p.add_argument("--ambient", type=float, default=0.3,
                   help="ambient scalar (SimpleObjectRenderer default)")
    p.add_argument("--scale-clamp", type=float, nargs=2, default=(0.5, 3.0),
                   metavar=("LO", "HI"),
                   help="clamp per-channel diagonal scale to this range")
    args = p.parse_args()
    if args.apply_ccm:
        apply(args.mesh, args.apply_ccm, args.out)
        return
    if not args.image or not args.mask:
        p.error("--image and --mask are required unless --apply-ccm is given")
    run(args.image, args.mask, args.mesh, args.out,
        width=args.width, height=args.height,
        fov=args.fov, cam_distance=args.cam_distance,
        alpha_thresh=args.alpha_thresh,
        intensity_mult=args.intensity_mult, ambient=args.ambient,
        scale_clamp=tuple(args.scale_clamp))


if __name__ == "__main__":
    main()
