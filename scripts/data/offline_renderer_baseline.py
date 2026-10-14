"""Mesh-only offline renderer (no gsplat) for baseline / sanity-check videos.

Renders the same robot + statics + objects as `offline_renderer.py`, but with
a programmatic ivory ground plane + sky-colored background instead of a
gsplat splat. Useful as a baseline to inspect geometry/poses without dragging
in a per-scene .ply.

Differences vs `offline_renderer.py`:
  - No `VisualizerSceneManager` / `--ply` / per-scene calibration.
  - No HSV jitter (no real-world ISP gap to bridge here).
  - No augmentation-transform pass (renders the original trajectory only).
  - Forces every static + object to render — including entries marked
    `render_in_image: False` in scene_config (those are flagged off because
    they're baked into the gsplat; the baseline has no gsplat to bake into).

Objects render with their raw SAM 3D meshes; fetch them first with
`scripts/download_assets.sh sources` (the tables come with the core assets).

Usage:
    cd scripts/data
    python offline_renderer_baseline.py --data-dir <episode-dir> [--episode 0000] [--video]
    # appearance variant (scene_config.py), e.g. the paper's apple/box re-skin:
    python offline_renderer_baseline.py --data-dir <episode-dir> --scene apple_to_box_turn
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYRENDER_OFFSCREEN", "1")

# pyrender 0.1.45 + OSMesa: read_color_buf hardcodes GL_RGB so RGBA flag is
# silently dropped. Pad alpha from depth (=255 where geometry was drawn).
if os.environ.get("PYOPENGL_PLATFORM") == "osmesa":
    import numpy as _np
    import pyrender as _pyr
    _orig_offscreen_render = _pyr.OffscreenRenderer.render
    def _render_with_alpha(self, scene, flags=0, seg_node_map=None):
        result = _orig_offscreen_render(self, scene, flags, seg_node_map)
        if flags & _pyr.constants.RenderFlags.RGBA and isinstance(result, tuple):
            color, depth = result
            if color.shape[-1] == 3:
                alpha = (depth > 0).astype(_np.uint8) * 255
                color = _np.dstack([color, alpha])
                result = (color, depth)
        return result
    _pyr.OffscreenRenderer.render = _render_with_alpha

import argparse, json, cv2, numpy as np, yourdfpy, glob, trimesh, time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
from renderer.robot_renderer import RobotMeshRenderer
from renderer.object_renderer import SimpleObjectRenderer
from scene_config import (get_scene, DEFAULT_SCENE, SCENES, _OBJECTS_DIR,
                          asset_group, missing_asset_message, require_core_assets)

URDF_PATH = str(Path(__file__).resolve().parents[2] / "robots/g1/urdf/g1_29dof_with_hand.urdf")

# Ivory floor, slightly cooler sky so the horizon line is visible.
IVORY_RGB = (245, 230, 200)
SKY_RGB   = (220, 225, 235)

IMG_EXT = "jpg"
IMG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]

_PNG_WRITE_QUEUE_LIMIT = 48


def _build_ground_mesh(size=20.0, color=IVORY_RGB):
    """Thin ivory slab centered at xy=0, top at z=0. Acts as the floor."""
    m = trimesh.creation.box(extents=[size, size, 0.02])
    m.apply_translation([0.0, 0.0, -0.01])
    rgba = np.array([color[0], color[1], color[2], 255], dtype=np.uint8)
    m.visual.face_colors = np.tile(rgba, (len(m.faces), 1))
    return m


def _resolve_mesh_path(entry):
    """Prefer the raw SAM 3D mesh, `sources/<base>/mesh_raw.glb`, over `<base>.glb`.

    The baseline is a no-gsplat A/B. Color refinement tones a mesh toward the
    iPhone photo's white balance, which only makes sense alongside the gsplat
    composite, so mesh-only renders use the unmodified SAM 3D vertex colors.
    The raw orange, plate, apple and box meshes come from
    `scripts/download_assets.sh sources`; a missing one is an error, not a
    silent fallback. Entities without a raw mesh in the asset manifest (the
    tables, from the core assets) render their `<base>.glb`, then the configured file.
    """
    root = _OBJECTS_DIR
    base_no_ext, _ = os.path.splitext(entry["mesh"])
    raw = os.path.join(root, "sources", base_no_ext, "mesh_raw.glb")
    if os.path.exists(raw):
        return raw
    if asset_group(raw):
        raise FileNotFoundError(missing_asset_message(raw))
    for path in (os.path.join(root, base_no_ext + ".glb"), os.path.join(root, entry["mesh"])):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(missing_asset_message(os.path.join(root, base_no_ext + ".glb")))


def _load_entity_mesh(entry, path=None):
    path = path or _resolve_mesh_path(entry)
    if path.endswith(".glb"):
        loaded = trimesh.load(path)
        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
        correction = np.eye(4, dtype=np.float32)
        correction[:3, :3] = R.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
        mesh.apply_transform(correction)
    else:
        mesh = trimesh.load(path, force="mesh")

    # Honor the SAM 3D mesh's own colors — PBR textures (table) AND
    # vertex/face colors (plate gray-blue, orange orange) both come from
    # the source GLB. Only fall back to scene_config rgba when the mesh
    # has no color info at all (untextured STLs would otherwise render grey).
    if "rgba" in entry and not _has_native_color(mesh):
        rgba = np.clip(np.asarray(entry["rgba"], dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
        if hasattr(mesh.visual, "face_colors"):
            mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))
    return mesh


def _has_native_color(mesh):
    # trimesh.visual.kind is the cheapest discriminator:
    #   TextureVisuals → "texture", ColorVisuals → "vertex"/"face"/None.
    # SAM 3D plate/orange ship vertex_colors (no PBR), so the old
    # baseColorTexture-only check used to drop them on the floor.
    vis = getattr(mesh, "visual", None)
    if vis is None:
        return False
    return getattr(vis, "kind", None) in ("texture", "vertex", "face")


def build_entity_renderers(scene_cfg, viewport_w, viewport_h):
    """One pyrender per static + object. Ignores render_in_image=False —
    the baseline has no gsplat to bake tables into, so every prop must
    render explicitly."""
    renderers = {}
    for entry in list(scene_cfg.get("statics", [])) + list(scene_cfg.get("objects", [])):
        name = entry["name"]
        mesh = _load_entity_mesh(entry)  # raises if no mesh file exists
        rndr = SimpleObjectRenderer(viewport_width=viewport_w, viewport_height=viewport_h)
        rndr.set_mesh(mesh, smooth=False)
        renderers[name] = (rndr, float(entry.get("scale", 1.0)))
    return renderers


def composite_by_depth(bg_rgb, fg_layers):
    """Depth-sort foreground layers, alpha-over onto a uniform bg."""
    H, W = bg_rgb.shape[:2]
    final = bg_rgb.copy()
    if not fg_layers:
        return final

    FAR = 999.0
    best_depth = np.full((H, W), FAR, dtype=np.float32)
    best_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for rgba, depth in fg_layers:
        depth = depth.squeeze()
        alpha = rgba[..., 3]
        vis = (alpha > 0) & (depth > 0.01)
        closer = vis & (depth < best_depth)
        best_depth[closer] = depth[closer]
        best_rgba[closer] = rgba[closer]

    alpha = best_rgba[..., 3]
    fg_vis = alpha > 0
    if fg_vis.any():
        opaque = alpha == 255
        final[opaque] = best_rgba[opaque, :3]
        boundary = fg_vis & ~opaque
        if boundary.any():
            fa = best_rgba[boundary, 3:4].astype(np.uint16)
            fg = best_rgba[boundary, :3].astype(np.uint16)
            bg = final[boundary].astype(np.uint16)
            final[boundary] = (fg + (bg * (255 - fa)) // 255).astype(np.uint8)
    return final


def _pose_to_T(pos_w, wxyz_w):
    if pos_w is None or wxyz_w is None:
        return None
    T = np.eye(4, dtype=np.float32)
    qw, qx, qy, qz = wxyz_w
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
    T[:3, 3] = np.asarray(pos_w, dtype=np.float32)
    return T


def render_episode(records, static_poses, robot_renderer, entity_renderers,
                   ground_renderer, ep_out_dir, camera_metadata, joint_names,
                   cameras=("d435",), lock_orient_names=None):
    base_W, base_H = 640, 480
    sky = np.array(SKY_RGB, dtype=np.uint8)

    t_start = time.perf_counter()
    writer = ThreadPoolExecutor(max_workers=8, thread_name_prefix="jpg")
    write_futures = []

    for frame_idx, record in enumerate(records):
        rob = record["robot"]
        T_base = np.eye(4, dtype=np.float32)
        qw, qx, qy, qz = rob["base_wxyz_w"]
        T_base[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        T_base[:3, 3] = np.asarray(rob["base_pos_w"], dtype=np.float32)

        object_T = {}
        for name, data in (record.get("objects") or {}).items():
            if name not in entity_renderers:
                continue
            T_w = _pose_to_T(data.get("pos_w"), data.get("wxyz_w"))
            if T_w is None:
                continue
            if lock_orient_names and name in lock_orient_names:
                T_w[:3, :3] = np.eye(3, dtype=T_w.dtype)
            object_T[name] = T_w

        joints_dict = dict(zip(joint_names, rob["joints"]))

        for cam_name, cam_data in record["cameras"].items():
            if cam_name not in cameras:
                continue
            T_cam = _pose_to_T(cam_data["pos_w"], cam_data["wxyz_w"])
            fov = float(camera_metadata.get(cam_name, {}).get("fov", 0.74))
            W, H = base_W, base_H

            bg = np.broadcast_to(sky, (H, W, 3)).copy()

            # Ground first (anchored to world identity).
            img_g, depth_g = ground_renderer.render(
                T_cam, np.eye(4, dtype=np.float32), 1.0, W, H, fov
            )
            img_rob, depth_rob = robot_renderer.render(joints_dict, T_base, T_cam, W, H, fov)

            fg_layers = [(img_g, depth_g), (img_rob, depth_rob)]
            for name, T_world in static_poses.items():
                if name not in entity_renderers:
                    continue
                rndr, scale = entity_renderers[name]
                img_e, depth_e = rndr.render(T_cam, T_world, scale, W, H, fov)
                fg_layers.append((img_e, depth_e))
            for name, T_world in object_T.items():
                rndr, scale = entity_renderers[name]
                img_e, depth_e = rndr.render(T_cam, T_world, scale, W, H, fov)
                fg_layers.append((img_e, depth_e))

            final_img = composite_by_depth(bg, fg_layers)

            out_path = os.path.join(ep_out_dir, f"{frame_idx:05d}_{cam_name}.{IMG_EXT}")
            bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            write_futures.append(writer.submit(cv2.imwrite, out_path, bgr, IMG_PARAMS))
            if len(write_futures) >= _PNG_WRITE_QUEUE_LIMIT:
                write_futures.pop(0).result()

        print(f"  Frame {frame_idx:05d}/{len(records)}", end="\r")

    print()
    for f in write_futures:
        f.result()
    writer.shutdown(wait=True)
    t = time.perf_counter() - t_start
    n = len(records)
    fps = n / t if t > 0 else 0.0
    print(f"  [Timing] {n} frames in {t:.2f}s -> {fps:.2f} fps "
          f"({1000.0 * t / max(n, 1):.1f} ms/frame)")


def _parse_scenario(scenario, override_scene=None):
    recorded = scenario.get("scene_name")
    scene_name = override_scene or recorded or DEFAULT_SCENE
    if scene_name not in SCENES:
        raise ValueError(f"Unknown scene '{scene_name}'; pass --scene with one of: "
                         f"{', '.join(sorted(SCENES))}")
    base = SCENES[scene_name].get("base_scene")
    if base and recorded and recorded != base:
        raise ValueError(f"Scene '{scene_name}' re-renders episodes recorded with '{base}', "
                         f"but this episode was recorded with '{recorded}'.")
    scene_cfg = get_scene(scene_name)
    static_poses = {}
    for s in scenario.get("statics") or []:
        T = _pose_to_T(s.get("pos_w"), s.get("wxyz_w"))
        if T is not None:
            static_poses[s["name"]] = T
    return scene_cfg, scene_name, static_poses, \
           scenario.get("camera_metadata", {}), scenario.get("joint_names", [])


def stitch_video(frames_dir, cam, ep_str, fps=20):
    frames = sorted(glob.glob(os.path.join(frames_dir, f"*_{cam}.{IMG_EXT}")))
    if not frames:
        return None
    first = cv2.imread(frames[0])
    H, W = first.shape[:2]
    mp4_path = os.path.join(frames_dir, f"episode_{ep_str}_{cam}.mp4")
    vw = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fp in frames:
        vw.write(cv2.imread(fp))
    vw.release()
    print(f"  Wrote {mp4_path} ({len(frames)} frames @ {fps}fps)")
    return mp4_path


def _enumerate_episode_files(data_dir, episode=None, episode_from=None):
    if episode is not None:
        path = os.path.join(data_dir, f"episode_{episode}.json")
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            return []
        return [path]
    robot_files = sorted(glob.glob(os.path.join(data_dir, "episode_[0-9][0-9][0-9][0-9].json")))
    if episode_from is not None:
        before = len(robot_files)
        robot_files = [
            p for p in robot_files
            if int(os.path.basename(p).replace("episode_", "").replace(".json", "")) >= episode_from
        ]
        print(f"[BaselineRender] --episode-from {episode_from}: {before} -> {len(robot_files)} files")
    return robot_files


def render_episode_files(robot_files, data_dir, scene_override=None,
                          out_dir=None, cameras=("d435",), video=False, fps=20,
                          worker_tag=""):
    """Render the given episode JSONs end-to-end with mesh-only baseline.

    Builds one robot_renderer + ground_renderer + per-scene entity_renderer
    cache per call, so a multi-process wrapper can call this in each worker
    and every process gets its own EGL/pyrender state."""
    if not robot_files:
        return
    output_root = out_dir or os.path.join(data_dir, "baseline_renders")
    os.makedirs(output_root, exist_ok=True)
    base_W, base_H = 640, 480

    tag = f"[{worker_tag}] " if worker_tag else ""
    print(f"{tag}Initializing baseline renderer (mesh-only, no gsplat)...")
    robot_renderer = RobotMeshRenderer(yourdfpy.URDF.load(URDF_PATH), URDF_PATH)
    ground_renderer = SimpleObjectRenderer(viewport_width=base_W, viewport_height=base_H)
    ground_renderer.set_mesh(_build_ground_mesh(size=20.0, color=IVORY_RGB), smooth=False)

    renderer_cache = {}

    for rob_file in robot_files:
        ep_str = os.path.basename(rob_file).replace("episode_", "").replace(".json", "")
        # Write under episode_XXXX/00/ so build_sonic_lerobot.py's aug-subdir
        # layout (default render_subdir="00") finds the frames.
        ep_out_dir = os.path.join(output_root, f"episode_{ep_str}", "00")
        os.makedirs(ep_out_dir, exist_ok=True)

        scenario_file = os.path.join(data_dir, f"episode_{ep_str}_scenario.json")
        if not os.path.exists(scenario_file):
            print(f"{tag}  Missing {scenario_file}; skipping")
            continue
        with open(scenario_file) as f:
            scenario = json.load(f)

        scene_cfg, scene_name, static_poses, camera_metadata, joint_names = \
            _parse_scenario(scenario, override_scene=scene_override)

        if scene_name not in renderer_cache:
            print(f"{tag}  Building entity renderers for scene '{scene_name}'...")
            renderer_cache[scene_name] = build_entity_renderers(scene_cfg, base_W, base_H)
        entity_renderers = renderer_cache[scene_name]
        lock_orient_names = {
            e["name"] for e in scene_cfg.get("objects", [])
            if e.get("lock_world_orientation")
        }

        with open(rob_file) as f:
            records = json.load(f)

        print(f"{tag}--- Episode {ep_str}: {len(records)} frames, scene='{scene_name}' ---")
        render_episode(records, static_poses, robot_renderer, entity_renderers,
                       ground_renderer, ep_out_dir, camera_metadata, joint_names,
                       cameras=cameras, lock_orient_names=lock_orient_names)

        if video:
            for cam in cameras:
                stitch_video(ep_out_dir, cam, ep_str, fps=fps)

    print(f"\n{tag}Baseline rendering complete!")


def check_scene_assets(scene_name):
    """Fail fast (before any GL work) if a core asset is missing, or a --scene
    is unknown or misses a mesh."""
    require_core_assets()
    if scene_name is None:
        return
    if scene_name not in SCENES:
        raise SystemExit(f"Unknown --scene '{scene_name}'. Known scenes: {', '.join(sorted(SCENES))}")
    cfg = SCENES[scene_name]
    for entry in list(cfg.get("statics", [])) + list(cfg.get("objects", [])):
        try:
            _resolve_mesh_path(entry)
        except FileNotFoundError as e:
            raise SystemExit(str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Episode dir (reads episode_*.json, writes baseline_renders/).")
    parser.add_argument("--episode", type=str, default=None,
                        help="Render a single episode, e.g. '0000'.")
    parser.add_argument("--episode-from", type=int, default=None,
                        help="Skip episodes below this index (ignored when --episode is set).")
    parser.add_argument("--scene", type=str, default=None,
                        help="Render with this scene instead of the recorded one, e.g. an "
                             "appearance variant such as apple_to_box_turn (scene_config.py).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output root (default <data-dir>/baseline_renders/).")
    parser.add_argument("--mid360", action="store_true",
                        help="Also render mid360 camera.")
    parser.add_argument("--video", action="store_true",
                        help="Stitch the rendered JPGs into an MP4.")
    parser.add_argument("--fps", type=int, default=20,
                        help="FPS for stitched MP4 (default 20).")
    args = parser.parse_args()
    check_scene_assets(args.scene)

    cameras = ("d435", "mid360") if args.mid360 else ("d435",)

    robot_files = _enumerate_episode_files(
        args.data_dir, episode=args.episode, episode_from=args.episode_from,
    )
    if not robot_files:
        raise SystemExit(f"Error: No episode data found in {args.data_dir}")

    render_episode_files(
        robot_files, args.data_dir,
        scene_override=args.scene, out_dir=args.out_dir,
        cameras=cameras, video=args.video, fps=args.fps,
    )


if __name__ == "__main__":
    main()
