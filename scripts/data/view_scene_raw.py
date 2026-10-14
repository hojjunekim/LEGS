#!/usr/bin/env python
"""Load a scene's raw .obj mesh and raw .ply gsplat side-by-side in Viser so
you can rotate/zoom and check if they're aligned in the raw frame.

A transform-controls gizmo (and slider folder) drives the OBJ pose. Use the
Save Calibration button to persist the mesh->gsplat transform to
data/gsplat/{scene}/mesh_to_gsplat.json:

  { "translation": [tx, ty, tz],
    "rotation_rpy_deg": [r, p, y],
    "scale": s }

meaning v_ply = R(rpy_xyz) @ (v_obj * scale) + translation. Downstream
(extract_scene_table.py) can chain this with the existing default.json so
the cropped mesh lands at the same world XYZ as the gsplat.

  python scripts/data/view_scene_raw.py --scene <name>

Needs data/gsplat/<name>.ply plus the capture app's raw mesh export at
data/gsplat/<name>/3DModel.obj (the scan mesh is not shipped with the repo).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

_REPO = Path(__file__).resolve().parents[2]
GSPLAT_DIR = _REPO / "data" / "gsplat"


SCENE_CROP_DEFAULTS = {
    "wood": {"x_range": [-0.5, 2.0], "y_range": [-0.5, 2.0], "z_range": [0.1, 0.8]},
}


def load_transform(p: Path):
    if not p.exists():
        return np.zeros(3, np.float32), np.array([1, 0, 0, 0], np.float32), 1.0
    d = json.loads(p.read_text())
    pos = np.asarray(d.get("translation", [0, 0, 0]), dtype=np.float32)
    rpy = np.asarray(d.get("rotation_rpy_deg", [0, 0, 0]), dtype=np.float64)
    scale = float(d.get("scale", 1.0))
    q = R.from_euler("xyz", rpy, degrees=True).as_quat()  # xyzw
    wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return pos, wxyz, scale


def load_crop(p: Path, scene: str):
    """Crop precedence: saved file `crop` field > scene default > wide-open."""
    defaults = SCENE_CROP_DEFAULTS.get(scene, {})
    x = defaults.get("x_range", [-5.0, 5.0])
    y = defaults.get("y_range", [-5.0, 5.0])
    z = defaults.get("z_range", [-1.0, 3.0])
    min_faces = defaults.get("min_faces", 0)
    top_n = defaults.get("top_n", 0)  # 0 = no cap
    if p.exists():
        c = json.loads(p.read_text()).get("crop")
        if isinstance(c, dict):
            x = list(c.get("x_range", x))
            y = list(c.get("y_range", y))
            z = list(c.get("z_range", z))
            min_faces = int(c.get("min_faces", min_faces))
            top_n = int(c.get("top_n", top_n))
    return x, y, z, min_faces, top_n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True,
                    help="splat name: reads data/gsplat/<scene>.ply and <scene>/3DModel.obj")
    ap.add_argument("--ply-clip", type=float, default=8.0,
                    help="drop PLY points outside +/- this (m); 0 to keep everything")
    ap.add_argument("--ply-points", type=int, default=200_000,
                    help="max PLY points sent to viser (downsampled)")
    ap.add_argument("--z-min", type=float, default=0.10,
                    help="drop PLY points below this world z (m); default 0.10")
    ap.add_argument("--x-range", type=float, nargs=2, default=None,
                    metavar=("XMIN", "XMAX"),
                    help="crop PLY + mesh to this world x range (m)")
    ap.add_argument("--y-range", type=float, nargs=2, default=None,
                    metavar=("YMIN", "YMAX"),
                    help="crop PLY + mesh to this world y range (m)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    obj_path = GSPLAT_DIR / args.scene / "3DModel.obj"
    ply_path = GSPLAT_DIR / f"{args.scene}.ply"
    calib_path = GSPLAT_DIR / args.scene / "mesh_to_gsplat.json"
    for p in (ply_path, obj_path):
        if not p.exists():
            raise SystemExit(f"{p.relative_to(_REPO)} not found (needs the splat and the "
                             "capture's raw 3DModel.obj export)")

    print(f"[load] mesh {obj_path}")
    mesh = trimesh.load(str(obj_path), process=False, force="mesh")
    print(f"       raw  verts: {len(mesh.vertices):,}  faces: {len(mesh.faces):,}")
    mesh.merge_vertices()  # weld duplicates so connected-component split works
    obj_v_raw = np.asarray(mesh.vertices, dtype=np.float32)
    obj_f = np.asarray(mesh.faces, dtype=np.int32)
    print(f"       weld verts: {len(obj_v_raw):,}  faces: {len(obj_f):,}")

    print(f"[load] ply  {ply_path}")
    ply = trimesh.load(str(ply_path), process=False)
    ply_v_all = np.asarray(ply.vertices, dtype=np.float32)
    print(f"       points: {len(ply_v_all):,}")
    if args.ply_clip > 0:
        ply_v_all = ply_v_all[np.all(np.abs(ply_v_all) < args.ply_clip, axis=1)]
        print(f"       after +/-{args.ply_clip}m clip: {len(ply_v_all):,}")

    init_pos, init_wxyz, init_scale = load_transform(calib_path)
    if calib_path.exists():
        print(f"[calib] loaded existing {calib_path}")
    else:
        print(f"[calib] no existing {calib_path} — starting from identity")

    # Resolve crop bounds (CLI > saved file > scene default) and pre-crop
    # OBJ + PLY using the initial calibration. The full mesh is too big to
    # hold/rebuild every frame; the sliders later refine within this subset.
    saved_x, saved_y, saved_z, saved_min_faces, saved_top_n = load_crop(
        calib_path, args.scene
    )
    init_xmin, init_xmax = args.x_range if args.x_range is not None else saved_x
    init_ymin, init_ymax = args.y_range if args.y_range is not None else saved_y
    init_zmin = args.z_min if args.z_min is not None else saved_z[0]
    init_zmax = saved_z[1]

    R_init = R.from_quat(
        [init_wxyz[1], init_wxyz[2], init_wxyz[3], init_wxyz[0]]
    ).as_matrix().astype(np.float32)
    obj_v_world = (obj_v_raw * init_scale) @ R_init.T + init_pos
    keep_v = (
        (obj_v_world[:, 0] >= init_xmin) & (obj_v_world[:, 0] <= init_xmax) &
        (obj_v_world[:, 1] >= init_ymin) & (obj_v_world[:, 1] <= init_ymax) &
        (obj_v_world[:, 2] >= init_zmin) & (obj_v_world[:, 2] <= init_zmax)
    )
    keep_f = keep_v[obj_f].all(axis=1)
    old_to_new = -np.ones(len(obj_v_raw), dtype=np.int32)
    old_to_new[keep_v] = np.arange(int(keep_v.sum()), dtype=np.int32)
    obj_v_raw = obj_v_raw[keep_v]
    obj_f = old_to_new[obj_f[keep_f]]
    print(
        f"[crop] mesh -> verts {len(obj_v_raw):,}  faces {len(obj_f):,}  "
        f"(box x={init_xmin:.2f}..{init_xmax:.2f}, "
        f"y={init_ymin:.2f}..{init_ymax:.2f}, "
        f"z={init_zmin:.2f}..{init_zmax:.2f})"
    )

    ply_mask = (
        (ply_v_all[:, 0] >= init_xmin) & (ply_v_all[:, 0] <= init_xmax) &
        (ply_v_all[:, 1] >= init_ymin) & (ply_v_all[:, 1] <= init_ymax) &
        (ply_v_all[:, 2] >= init_zmin) & (ply_v_all[:, 2] <= init_zmax)
    )
    ply_v_all = ply_v_all[ply_mask]
    print(f"[crop] ply  -> points {len(ply_v_all):,}")

    server = viser.ViserServer(port=args.port)

    # World origin axes for orientation reference
    server.scene.add_frame("/origin", axes_length=0.5, axes_radius=0.01)

    # PLY (orange) — re-built whenever crop sliders change
    ply_handle = {"h": None}

    # OBJ parent frame (gizmo drives this); mesh + its own axes live under it
    obj_frame = server.scene.add_frame(
        "/obj_calib", position=init_pos, wxyz=init_wxyz,
        axes_length=0.4, axes_radius=0.008,
    )

    # Mesh handle — rebuilt on scale / gizmo / crop slider updates
    mesh_handle = {"h": None}

    # Gizmo — separate node, sync'd with obj_frame and sliders below
    gizmo = server.scene.add_transform_controls(
        "/obj_gizmo", scale=0.5, position=init_pos, wxyz=init_wxyz,
    )

    with server.gui.add_folder("Mesh→GSplat Calibration", expand_by_default=True):
        show_gizmo = server.gui.add_checkbox("Show Gizmo", initial_value=True)
        scale_slider = server.gui.add_slider("Scale", min=0.1, max=10.0, step=0.005,
                                             initial_value=init_scale)
        tx_slider = server.gui.add_slider("X (m)", min=-5.0, max=5.0, step=0.005,
                                          initial_value=float(init_pos[0]))
        ty_slider = server.gui.add_slider("Y (m)", min=-5.0, max=5.0, step=0.005,
                                          initial_value=float(init_pos[1]))
        tz_slider = server.gui.add_slider("Z (m)", min=-5.0, max=5.0, step=0.005,
                                          initial_value=float(init_pos[2]))
        init_rpy = R.from_quat(
            [init_wxyz[1], init_wxyz[2], init_wxyz[3], init_wxyz[0]]
        ).as_euler("xyz", degrees=True)
        roll_slider = server.gui.add_slider("Roll (deg)", min=-180.0, max=180.0,
                                            step=0.25, initial_value=float(init_rpy[0]))
        pitch_slider = server.gui.add_slider("Pitch (deg)", min=-180.0, max=180.0,
                                             step=0.25, initial_value=float(init_rpy[1]))
        yaw_slider = server.gui.add_slider("Yaw (deg)", min=-180.0, max=180.0,
                                           step=0.25, initial_value=float(init_rpy[2]))
        reset_btn = server.gui.add_button("Reset")
        print_btn = server.gui.add_button("Print")
        save_btn = server.gui.add_button("Save Calibration")

    # Crop sliders (init from pre-crop bounds computed above)
    with server.gui.add_folder("Crop (world m)", expand_by_default=True):
        crop_enable = server.gui.add_checkbox("Crop enabled", initial_value=True)
        xmin_slider = server.gui.add_slider("X min", min=-5.0, max=5.0, step=0.01,
                                            initial_value=float(init_xmin))
        xmax_slider = server.gui.add_slider("X max", min=-5.0, max=5.0, step=0.01,
                                            initial_value=float(init_xmax))
        ymin_slider = server.gui.add_slider("Y min", min=-5.0, max=5.0, step=0.01,
                                            initial_value=float(init_ymin))
        ymax_slider = server.gui.add_slider("Y max", min=-5.0, max=5.0, step=0.01,
                                            initial_value=float(init_ymax))
        zmin_slider = server.gui.add_slider("Z min", min=-1.0, max=3.0, step=0.01,
                                            initial_value=float(init_zmin))
        zmax_slider = server.gui.add_slider("Z max", min=-1.0, max=5.0, step=0.01,
                                            initial_value=float(init_zmax))
        ply_size_slider = server.gui.add_slider("PLY point size", min=0.001, max=0.05,
                                                step=0.001, initial_value=0.015)
        min_faces_slider = server.gui.add_slider(
            "Min faces per cluster", min=0, max=20000, step=50,
            initial_value=int(saved_min_faces),
        )
        top_n_slider = server.gui.add_slider(
            "Keep top N clusters (0=no cap)", min=0, max=10000, step=10,
            initial_value=int(saved_top_n),
        )
        save_crop_btn = server.gui.add_button("Save Crop")
        bake_name_input = server.gui.add_text(
            "Bake STL name", initial_value=f"{args.scene}_scene",
        )
        bake_btn = server.gui.add_button("Bake to MuJoCo STL")

    sync = {"flag": False}
    _ply_rng = np.random.default_rng(0)
    _mesh_debug = {"first": True}

    def _crop_bounds():
        if not bool(crop_enable.value):
            return (-1e9, 1e9, -1e9, 1e9, -1e9, 1e9)
        return (
            float(xmin_slider.value), float(xmax_slider.value),
            float(ymin_slider.value), float(ymax_slider.value),
            float(zmin_slider.value), float(zmax_slider.value),
        )

    def _rebuild_ply():
        xmin, xmax, ymin, ymax, zmin, zmax = _crop_bounds()
        v = ply_v_all
        mask = (
            (v[:, 0] >= xmin) & (v[:, 0] <= xmax) &
            (v[:, 1] >= ymin) & (v[:, 1] <= ymax) &
            (v[:, 2] >= zmin) & (v[:, 2] <= zmax)
        )
        v = v[mask]
        if len(v) > args.ply_points:
            idx = _ply_rng.choice(len(v), args.ply_points, replace=False)
            v = v[idx]
        if ply_handle["h"] is not None:
            ply_handle["h"].remove()
        ply_handle["h"] = server.scene.add_point_cloud(
            "/ply", points=v,
            colors=np.tile(np.array([[240, 130, 30]], dtype=np.uint8), (len(v), 1)),
            point_size=float(ply_size_slider.value),
        )

    def _rebuild_mesh():
        scale = float(scale_slider.value)
        pos = np.asarray(gizmo.position, dtype=np.float32)
        w = gizmo.wxyz
        R_world = R.from_quat(
            [w[1], w[2], w[3], w[0]]
        ).as_matrix().astype(np.float32)
        v_world = (obj_v_raw * scale) @ R_world.T + pos
        xmin, xmax, ymin, ymax, zmin, zmax = _crop_bounds()
        keep_v = (
            (v_world[:, 0] >= xmin) & (v_world[:, 0] <= xmax) &
            (v_world[:, 1] >= ymin) & (v_world[:, 1] <= ymax) &
            (v_world[:, 2] >= zmin) & (v_world[:, 2] <= zmax)
        )
        if keep_v.all():
            v_use, f_use = obj_v_raw, obj_f
        else:
            keep_f = keep_v[obj_f].all(axis=1)
            old_to_new = -np.ones(len(obj_v_raw), dtype=np.int32)
            old_to_new[keep_v] = np.arange(int(keep_v.sum()), dtype=np.int32)
            v_use = obj_v_raw[keep_v]
            f_use = old_to_new[obj_f[keep_f]]

        if _mesh_debug["first"]:
            print(f"[mesh] after box crop: verts {len(v_use):,} faces {len(f_use):,}",
                  flush=True)

        # Connected-component noise filter (general, not table-specific):
        # split crop result, drop tiny clusters, keep top N by face count.
        # top_n == 0 means no cap.
        min_faces = int(min_faces_slider.value)
        top_n = int(top_n_slider.value)
        if len(f_use) > 0 and (min_faces > 0 or top_n > 0):
            sub = trimesh.Trimesh(vertices=v_use, faces=f_use, process=False)
            comps = sub.split(only_watertight=False)
            sizes = sorted([len(c.faces) for c in comps], reverse=True)
            if _mesh_debug["first"]:
                print(f"[mesh] split: {len(comps)} clusters; top-10 face counts: "
                      f"{sizes[:10]}", flush=True)
            comps = [c for c in comps if len(c.faces) >= min_faces]
            comps.sort(key=lambda c: len(c.faces), reverse=True)
            if top_n > 0:
                comps = comps[:top_n]
            if comps:
                merged = trimesh.util.concatenate(comps)
                v_use = np.asarray(merged.vertices, dtype=np.float32)
                f_use = np.asarray(merged.faces, dtype=np.int32)
            else:
                v_use = np.zeros((0, 3), dtype=np.float32)
                f_use = np.zeros((0, 3), dtype=np.int32)

        if _mesh_debug["first"]:
            print(f"[mesh] final: verts {len(v_use):,} faces {len(f_use):,}",
                  flush=True)
            _mesh_debug["first"] = False

        if mesh_handle["h"] is not None:
            mesh_handle["h"].remove()
            mesh_handle["h"] = None
        if len(f_use) == 0:
            return
        mesh_handle["h"] = server.scene.add_mesh_simple(
            "/obj_calib/mesh",
            vertices=(v_use * scale).astype(np.float32),
            faces=f_use.astype(np.int32),
            color=(60, 110, 220),
            opacity=0.65,
            flat_shading=True,
        )

    def _rebuild_all():
        _rebuild_ply()
        _rebuild_mesh()

    _rebuild_all()

    for s in (xmin_slider, xmax_slider, ymin_slider, ymax_slider,
              zmin_slider, zmax_slider):
        s.on_update(lambda _: _rebuild_all())
    crop_enable.on_update(lambda _: _rebuild_all())
    ply_size_slider.on_update(lambda _: _rebuild_ply())
    for s in (min_faces_slider, top_n_slider):
        s.on_update(lambda _: _rebuild_mesh())

    def _apply_pose():
        pos = gizmo.position
        obj_frame.position = pos
        obj_frame.wxyz = gizmo.wxyz

    def _push_sliders_from_gizmo():
        sync["flag"] = True
        try:
            tx_slider.value = float(gizmo.position[0])
            ty_slider.value = float(gizmo.position[1])
            tz_slider.value = float(gizmo.position[2])
            w = gizmo.wxyz
            rpy = R.from_quat([w[1], w[2], w[3], w[0]]).as_euler("xyz", degrees=True)
            roll_slider.value = float(rpy[0])
            pitch_slider.value = float(rpy[1])
            yaw_slider.value = float(rpy[2])
        finally:
            sync["flag"] = False

    def _push_gizmo_from_sliders():
        if sync["flag"]:
            return
        pos = np.array([tx_slider.value, ty_slider.value, tz_slider.value],
                       dtype=np.float32)
        q = R.from_euler("xyz",
                         [roll_slider.value, pitch_slider.value, yaw_slider.value],
                         degrees=True).as_quat()
        wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        gizmo.position = pos
        gizmo.wxyz = wxyz
        _apply_pose()

    @gizmo.on_update
    def _(_):
        _apply_pose()
        _push_sliders_from_gizmo()
        if bool(crop_enable.value):
            _rebuild_mesh()

    for s in (tx_slider, ty_slider, tz_slider,
              roll_slider, pitch_slider, yaw_slider):
        s.on_update(lambda _: _push_gizmo_from_sliders())

    @scale_slider.on_update
    def _(_):
        _rebuild_mesh()

    @show_gizmo.on_update
    def _(_):
        gizmo.visible = bool(show_gizmo.value)

    @reset_btn.on_click
    def _(_):
        sync["flag"] = True
        try:
            tx_slider.value = ty_slider.value = tz_slider.value = 0.0
            roll_slider.value = pitch_slider.value = yaw_slider.value = 0.0
            scale_slider.value = 1.0
        finally:
            sync["flag"] = False
        gizmo.position = np.zeros(3, np.float32)
        gizmo.wxyz = np.array([1, 0, 0, 0], np.float32)
        _apply_pose()
        _rebuild_mesh()

    def _current_calib():
        pos = gizmo.position.tolist()
        w = gizmo.wxyz
        rpy = R.from_quat([w[1], w[2], w[3], w[0]]).as_euler("xyz", degrees=True).tolist()
        return {
            "translation": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rotation_rpy_deg": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
            "scale": float(scale_slider.value),
        }

    @print_btn.on_click
    def _(_):
        c = _current_calib()
        print(f"[calib] translation: {c['translation']}")
        print(f"        rotation rpy deg: {c['rotation_rpy_deg']}")
        print(f"        scale: {c['scale']}")

    @save_btn.on_click
    def _(_):
        c = _current_calib()
        existing = json.loads(calib_path.read_text()) if calib_path.exists() else {}
        existing.update(c)
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        calib_path.write_text(json.dumps(existing, indent=2))
        print(f"[calib] saved -> {calib_path}")

    @bake_btn.on_click
    def _(_):
        try:
            scale = float(scale_slider.value)
            pos = np.asarray(gizmo.position, dtype=np.float32)
            w = gizmo.wxyz
            R_world = R.from_quat(
                [w[1], w[2], w[3], w[0]]
            ).as_matrix().astype(np.float32)
            v_world = (obj_v_raw * scale) @ R_world.T + pos
            xmin, xmax, ymin, ymax, zmin, zmax = _crop_bounds()
            keep_v = (
                (v_world[:, 0] >= xmin) & (v_world[:, 0] <= xmax) &
                (v_world[:, 1] >= ymin) & (v_world[:, 1] <= ymax) &
                (v_world[:, 2] >= zmin) & (v_world[:, 2] <= zmax)
            )
            if keep_v.all():
                v_use, f_use = obj_v_raw, obj_f
            else:
                keep_f = keep_v[obj_f].all(axis=1)
                old_to_new = -np.ones(len(obj_v_raw), dtype=np.int32)
                old_to_new[keep_v] = np.arange(int(keep_v.sum()), dtype=np.int32)
                v_use = obj_v_raw[keep_v]
                f_use = old_to_new[obj_f[keep_f]]

            mf = int(min_faces_slider.value)
            tn = int(top_n_slider.value)
            if len(f_use) > 0 and (mf > 0 or tn > 0):
                sub = trimesh.Trimesh(vertices=v_use, faces=f_use, process=False)
                comps = sub.split(only_watertight=False)
                comps = [c for c in comps if len(c.faces) >= mf]
                comps.sort(key=lambda c: len(c.faces), reverse=True)
                if tn > 0:
                    comps = comps[:tn]
                if not comps:
                    print("[bake] no clusters survived filter — nothing to bake",
                          flush=True)
                    return
                merged = trimesh.util.concatenate(comps)
                v_use = np.asarray(merged.vertices, dtype=np.float32)
                f_use = np.asarray(merged.faces, dtype=np.int32)

            if len(f_use) == 0:
                print("[bake] empty mesh after crop — nothing to bake", flush=True)
                return

            v_world_out = (
                (v_use * scale) @ R_world.T + pos
            ).astype(np.float32)

            bake_name = bake_name_input.value.strip()
            if not bake_name:
                print("[bake] empty name — set 'Bake STL name' first", flush=True)
                return

            out_dir = _REPO / "data" / "objects"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_stl = out_dir / f"{bake_name}.stl"
            col_dir = out_dir / bake_name

            trimesh.Trimesh(v_world_out, f_use, process=False).export(
                str(out_stl), file_type="stl"
            )
            print(f"[bake] wrote {out_stl}  "
                  f"(verts {len(v_world_out):,}, faces {len(f_use):,})",
                  flush=True)

            existing_parts = list(col_dir.glob(f"{bake_name}_col_*.stl"))
            if existing_parts:
                print(f"[bake] WARNING: {len(existing_parts)} stale CoACD parts "
                      f"in {col_dir}/. Remove them so CoACD re-decomposes:\n"
                      f"        rm {col_dir}/{bake_name}_col_*.stl",
                      flush=True)

            print("[bake] next: add this static to scene_config.py:", flush=True)
            print(f'        {{"name": "{bake_name}", '
                  f'"mesh": "{bake_name}.stl",\n'
                  f'         "pos": [0, 0, 0], "scale": 1.0, '
                  f'"render_in_image": False,\n'
                  f'         "coacd_threshold": 0.1, '
                  f'"coacd_max_convex_hull": 64}},', flush=True)
        except Exception as e:
            import traceback
            print(f"[bake] FAILED: {e!r}", flush=True)
            traceback.print_exc()

    @save_crop_btn.on_click
    def _(_):
        try:
            crop = {
                "x_range": [float(xmin_slider.value), float(xmax_slider.value)],
                "y_range": [float(ymin_slider.value), float(ymax_slider.value)],
                "z_range": [float(zmin_slider.value), float(zmax_slider.value)],
                "min_faces": int(min_faces_slider.value),
                "top_n": int(top_n_slider.value),
            }
            existing = (
                json.loads(calib_path.read_text()) if calib_path.exists() else {}
            )
            existing["crop"] = crop
            calib_path.parent.mkdir(parents=True, exist_ok=True)
            calib_path.write_text(json.dumps(existing, indent=2))
            print(
                f"[crop] saved -> {calib_path}  (x={crop['x_range']}, "
                f"y={crop['y_range']}, z={crop['z_range']}, "
                f"min_faces={crop['min_faces']}, top_n={crop['top_n']})",
                flush=True,
            )
        except Exception as e:
            import traceback
            print(f"[crop] SAVE FAILED: {e!r}", flush=True)
            traceback.print_exc()

    print()
    print("OBJ (blue mesh) + PLY (orange points) loaded.")
    print(f"Open http://localhost:{args.port}  — drag the gizmo to align "
          f"OBJ onto PLY, then click Save.")
    print(f"Save target: {calib_path}")
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
