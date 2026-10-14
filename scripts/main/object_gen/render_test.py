"""Step 4: Render mesh.glb (and mesh_refined.glb) at a canonical D435-style view.

Uses `scripts/main/renderer/object_renderer.SimpleObjectRenderer` — the same
class `offline_renderer.build_entity_renderers` uses — so what you see here
is what the simulator will render once the mesh is dropped into
`data/objects/`.

Outputs (under <out_dir>):
    test_view.png             # raw mesh
    test_view_refined.png     # refined mesh (skipped if file missing)

Usage:
    python render_test.py \
        --mesh         data/objects/sam3d_runs/mug/mesh.glb \
        --mesh-refined data/objects/sam3d_runs/mug/mesh_refined_ccm.glb \
        --out          data/objects/sam3d_runs/mug
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYRENDER_OFFSCREEN", "1")

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R

_REPO_MAIN = Path(__file__).resolve().parents[1]
if str(_REPO_MAIN) not in sys.path:
    sys.path.insert(0, str(_REPO_MAIN))

from renderer.object_renderer import SimpleObjectRenderer  # noqa: E402


def _load_mesh(path: str, y_to_z: bool = True) -> trimesh.Trimesh:
    """Match offline_renderer.load_entity_mesh: .glb gets X-90 (Y-up -> Z-up)."""
    loaded = trimesh.load(path)
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"loaded {path} as {type(mesh).__name__}, expected Trimesh")
    if y_to_z and path.lower().endswith(".glb"):
        correction = np.eye(4, dtype=np.float32)
        correction[:3, :3] = R.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
        mesh.apply_transform(correction)
    return mesh


def _look_at_opencv(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV-convention camera pose: +Z forward, +Y down, +X right (cam->world)."""
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


def render_one(mesh_path: str, out_path: str, width: int, height: int,
               fov: float, cam_distance: float, bg_color=(255, 255, 255)) -> None:
    mesh = _load_mesh(mesh_path)
    aabb_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.apply_translation(-aabb_center)  # center at origin

    rndr = SimpleObjectRenderer(viewport_width=width, viewport_height=height)
    rndr.set_mesh(mesh, smooth=False)
    # Unlit: vertex colors pass through directly, matching the iPhone photo
    # (renderer's PBR lighting mutes the Reinhard-transferred colors otherwise).
    rndr.set_lighting(intensity_mult=0.0, ambient=1.0)

    eye = np.array([cam_distance, 0.0, 0.05], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Z-up world
    T_cam = _look_at_opencv(eye, target, up)
    T_obj = np.eye(4, dtype=np.float32)

    rgba, _ = rndr.render(T_cam, T_obj, scale=1.0, width=width, height=height, fov_rad=fov)

    # Composite over the requested background (PNGs with alpha look fine, but
    # a white background reads better when comparing to a real photo).
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    bg = np.full_like(rgb, bg_color, dtype=np.float32)
    blended = (rgb * a + bg * (1 - a)).astype(np.uint8)

    Image.fromarray(blended).save(out_path)
    print(f"  wrote {out_path}")


def run(out_dir: str, mesh_path: str | None = None,
        mesh_refined_path: str | None = None,
        width: int = 640, height: int = 480,
        fov: float = 0.74, cam_distance: float = 0.4) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    if mesh_path and os.path.exists(mesh_path):
        out_png = out / "test_view.png"
        render_one(mesh_path, str(out_png), width, height, fov, cam_distance)
        results["test_view"] = str(out_png)
    if mesh_refined_path and os.path.exists(mesh_refined_path):
        out_png = out / "test_view_refined.png"
        render_one(mesh_refined_path, str(out_png), width, height, fov, cam_distance)
        results["test_view_refined"] = str(out_png)
    if not results:
        raise FileNotFoundError(
            "no mesh files found to render — pass --mesh and/or --mesh-refined"
        )
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", default=None)
    p.add_argument("--mesh-refined", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fov", type=float, default=0.74,
                   help="vertical FOV in radians (default 0.74, matches D435)")
    p.add_argument("--cam-distance", type=float, default=0.4,
                   help="camera distance along +X looking at origin (meters)")
    args = p.parse_args()
    run(args.out, args.mesh, args.mesh_refined,
        args.width, args.height, args.fov, args.cam_distance)


if __name__ == "__main__":
    main()
