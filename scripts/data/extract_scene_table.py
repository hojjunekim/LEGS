#!/usr/bin/env python
"""Crop a table-sized region out of a gsplat scene mesh and write it as a
centered .obj usable as a MuJoCo static mesh.

Pipeline:
  1. Load data/gsplat/{scene}/3DModel.obj
  2. (optional, --align-to-ply) Load data/gsplat/{scene}.ply, compute
     PLY-OBJ median offset within the OBJ bbox, pre-shift OBJ verts so they
     align with the gsplat in the raw frame.
  3. Read data/gsplat/{scene}/{calib}.json  (translation + rotation_rpy_deg + scale)
  4. world_verts = translation + R(rpy_xyz) @ (v * scale)
  5. Keep verts inside the XY box (center +/- half_extent) and the Z slab
     (z_floor <= z <= z_ceiling). Drop faces that lose any vertex.
  6. Write verts as-is in sim world coords (no re-centering). MJ places the
     mesh with pos=[0, 0, 0] and the table lands exactly where it sat in
     the scan after the default.json transform.

Usage (paths are relative to the repo root):
  python scripts/data/extract_scene_table.py --scene <name> [--calib default]

Needs data/gsplat/<name>/3DModel.obj, the capture app's raw mesh export
(not shipped with the repo).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

_REPO = Path(__file__).resolve().parents[2]
GSPLAT_DIR = _REPO / "data" / "gsplat"


def load_calib(path: Path):
    data = json.loads(path.read_text())
    t = np.asarray(data["translation"], dtype=np.float64)
    rpy = np.asarray(data["rotation_rpy_deg"], dtype=np.float64)
    s = float(data.get("scale", 1.0))
    Rm = R.from_euler("xyz", rpy, degrees=True).as_matrix()
    return t, Rm, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True,
                    help="splat name: reads data/gsplat/<scene>/3DModel.obj")
    ap.add_argument("--calib", default="default",
                    help="calibration json stem under data/gsplat/{scene}/ "
                         "(default: 'default', the canonical per-scene calib)")
    ap.add_argument("--x-range", nargs=2, type=float, metavar=("XMIN", "XMAX"),
                    default=[0.2, 1.8],
                    help="X bounds of the active scene region, sim world coords")
    ap.add_argument("--y-range", nargs=2, type=float, metavar=("YMIN", "YMAX"),
                    default=[-0.5, 0.5],
                    help="Y bounds of the active scene region, sim world coords")
    ap.add_argument("--z-floor", type=float, default=0.05,
                    help="lower Z bound; verts below this are dropped (drops floor)")
    ap.add_argument("--z-ceiling", type=float, default=0.85,
                    help="upper Z bound; verts above this are dropped (drops walls)")
    ap.add_argument("--out-dir", default=str(_REPO / "data" / "objects" / "scene"))
    ap.add_argument("--out-name", default=None,
                    help="output basename (no ext); default {scene}_{calib}_table")
    ap.add_argument("--no-calib", action="store_true",
                    help="skip the default.json transform; crop/write verts in "
                         "the raw scene frame (use raw-frame ranges)")
    ap.add_argument("--align-to-ply", action="store_true",
                    help="pre-shift OBJ verts by (PLY median - OBJ median) so the "
                         "mesh lands at the same world XYZ as the gsplat after "
                         "default.json (uses PLY points inside the OBJ bbox so "
                         "far-field gaussians don't bias the offset)")
    args = ap.parse_args()

    gs_root = GSPLAT_DIR / args.scene
    obj_path = gs_root / "3DModel.obj"
    calib_path = gs_root / f"{args.calib}.json"
    for p in [obj_path] + ([] if args.no_calib else [calib_path]):
        if not p.exists():
            raise SystemExit(f"{p.relative_to(_REPO)} not found")

    print(f"[load] mesh  {obj_path}")
    mesh = trimesh.load(str(obj_path), process=False, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    print(f"       raw verts: {len(verts):,}  faces: {len(faces):,}")

    if args.align_to_ply:
        ply_path = GSPLAT_DIR / f"{args.scene}.ply"
        print(f"[align] ply  {ply_path}")
        ply = trimesh.load(str(ply_path), process=False)
        ply_v = np.asarray(ply.vertices, dtype=np.float64)
        bb_lo = verts.min(axis=0)
        bb_hi = verts.max(axis=0)
        in_box = np.all((ply_v >= bb_lo) & (ply_v <= bb_hi), axis=1)
        n_in = int(in_box.sum())
        if n_in < 1000:
            raise SystemExit(f"too few PLY points ({n_in}) inside OBJ bbox to estimate offset")
        ply_med = np.median(ply_v[in_box], axis=0)
        obj_med = np.median(verts, axis=0)
        offset = ply_med - obj_med
        print(f"       PLY median (in OBJ bbox, n={n_in:,}): {np.round(ply_med, 4).tolist()}")
        print(f"       OBJ median:                          {np.round(obj_med, 4).tolist()}")
        print(f"       applying offset (PLY - OBJ):         {np.round(offset, 4).tolist()}")
        verts = verts + offset

    if args.no_calib:
        print("[calib] SKIPPED (--no-calib); crop ranges interpreted in raw scene frame")
        v_world = verts
    else:
        print(f"[load] calib {calib_path}")
        t, Rm, s = load_calib(calib_path)
        print(f"       translation: {np.round(t, 4).tolist()}")
        print(f"       rotation rpy deg: "
              f"{np.round(R.from_matrix(Rm).as_euler('xyz', degrees=True), 4).tolist()}")
        print(f"       scale: {s}")
        v_world = (Rm @ (verts.T * s)).T + t

    frame = "raw scene" if args.no_calib else "sim world"
    print(f"       {frame} bbox  "
          f"x=[{v_world[:,0].min():+.3f},{v_world[:,0].max():+.3f}]  "
          f"y=[{v_world[:,1].min():+.3f},{v_world[:,1].max():+.3f}]  "
          f"z=[{v_world[:,2].min():+.3f},{v_world[:,2].max():+.3f}]")

    x_min, x_max = args.x_range
    y_min, y_max = args.y_range
    keep_v = (
        (v_world[:, 0] >= x_min) & (v_world[:, 0] <= x_max)
        & (v_world[:, 1] >= y_min) & (v_world[:, 1] <= y_max)
        & (v_world[:, 2] >= args.z_floor) & (v_world[:, 2] <= args.z_ceiling)
    )
    n_kept = int(keep_v.sum())
    print(f"[crop] kept {n_kept:,} / {len(keep_v):,} verts "
          f"({100 * n_kept / len(keep_v):.1f}%)")
    if n_kept == 0:
        raise SystemExit("crop box is empty — adjust --x-range / --y-range / --z-* "
                         "after checking the world bbox above")

    keep_f = keep_v[faces].all(axis=1)
    print(f"       kept {int(keep_f.sum()):,} / {len(keep_f):,} faces "
          f"({100 * keep_f.sum() / len(keep_f):.1f}%)")

    new_idx = -np.ones(len(keep_v), dtype=np.int64)
    new_idx[keep_v] = np.arange(n_kept)
    new_faces = new_idx[faces[keep_f]]
    new_verts = v_world[keep_v].copy()  # verts stay in sim world coords

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_tag = "raw" if args.no_calib else args.calib
    stem = args.out_name or f"{args.scene}_{calib_tag}_table"
    out_obj = out_dir / f"{stem}.obj"
    out_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    out_mesh.export(str(out_obj))

    bbox_min = new_verts.min(axis=0)
    bbox_max = new_verts.max(axis=0)
    size = bbox_max - bbox_min
    print(f"[write] {out_obj}  ({len(new_verts):,} verts, {len(new_faces):,} faces)")
    print(f"        {frame} bbox  "
          f"x=[{bbox_min[0]:+.3f},{bbox_max[0]:+.3f}]  "
          f"y=[{bbox_min[1]:+.3f},{bbox_max[1]:+.3f}]  "
          f"z=[{bbox_min[2]:+.3f},{bbox_max[2]:+.3f}]")
    print(f"        size xyz (m): {np.round(size, 4).tolist()}")
    print()
    print("Suggested MuJoCo placement (scene_config.py):")
    print('  "name": "scene_table",')
    print(f'  "mesh": "{out_obj.name}",')
    if args.no_calib:
        print('  "pos": [0.0, 0.0, 0.0],   # verts in raw scene frame — apply your own pos/rpy in MJ')
    else:
        print('  "pos": [0.0, 0.0, 0.0],   # verts already in sim world coords')
    print('  "rpy": [0, 0, 0],')
    print('  "scale": 1.0,')


if __name__ == "__main__":
    main()
