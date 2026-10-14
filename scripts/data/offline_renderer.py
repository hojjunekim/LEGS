import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYRENDER_OFFSCREEN"] = "1"

import argparse, json, cv2, numpy as np, yourdfpy, glob, trimesh, torch, time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
from utils.gs_rendering_utils import VisualizerSceneManager, D435_REAL_K
from renderer.robot_renderer import RobotMeshRenderer
from renderer.object_renderer import SimpleObjectRenderer
from renderer.main_render_pipeline import MainRenderPipeline
from renderer.iphone_to_realsense import calibrate_iphone_to_realsense
from scene_config import (get_scene, DEFAULT_SCENE, SCENES, _OBJECTS_DIR,
                          asset_group, missing_asset_message, require_core_assets)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = "collected_data"  # default; override with --data-dir
URDF_PATH = str(_REPO_ROOT / "robots/g1/urdf/g1_29dof_with_hand.urdf")
# The paper's LEGS datasets are rendered over the wood-table splat
# (scripts/download_assets.sh paper); flightroom comes with the core assets.
DEFAULT_MAP = "wood"
DEFAULT_PLY = str(_REPO_ROOT / "data" / "gsplat" / f"{DEFAULT_MAP}.ply")

# JPEG q=95 — visually lossless for training data, ~10x faster encode than
# PNG-level-1 and ~5x smaller files. Flip IMG_EXT/IMG_PARAMS together to swap
# back to PNG (cv2.IMWRITE_PNG_COMPRESSION level 1 = fastest lossless).
IMG_EXT = "jpg"
IMG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]

# Pipeline order per frame:
#   1. render gsplat (iPhone domain, since the scene was iPhone-captured)
#   2. render mesh layers (iPhone domain — SAM3D meshes are color-calibrated to
#      iPhone via scripts/main/object_gen/color_refine_ccm.py)
#   3. composite by depth → iPhone-domain image
#   4. apply iphone2realsense CCM → RealSense-domain image
#   5. apply single HSV jitter in RealSense domain for domain randomization
APPLY_IPHONE_TO_REALSENSE = True
# Per-episode HSV jitter for sim2real bridging. One sample per render_episode
# call (so original episode + each augmentation get independent jitter), held
# constant across all frames. Applied to the final RealSense-domain composite.
# Off by default since the HDR-refit mesh + iphone2realsense CCM already covers
# the sim2real color gap; re-enable with --hsv-jitter when stacking on top.
ENABLE_HSV_JITTER = False
FINAL_HSV_RANGES = {"hue_deg": (-15.0, 15.0), "sat": (0.7, 1.3), "val": (0.85, 1.1)}

# Optional environment overrides for quick checks (all unset by default).
# MAX_FRAMES=N: render only the first N frames of each episode.
# FRAME_IDX=N: render only frame N of each episode.
# OUT_SUBDIR=DIR: write under <data-dir>/DIR/ instead of <data-dir>/final_renders/
#   (an absolute DIR is used as is).
# SKIP_AUG=1: skip the augmented trajectories (implied by MAX_FRAMES / FRAME_IDX).
_max_frames_env = os.environ.get("MAX_FRAMES", "").strip()
MAX_FRAMES = int(_max_frames_env) if _max_frames_env else None
_frame_idx_env = os.environ.get("FRAME_IDX", "").strip()
FRAME_IDX = int(_frame_idx_env) if _frame_idx_env else None
OUT_SUBDIR = os.environ.get("OUT_SUBDIR", "final_renders")
SKIP_AUG = (
    os.environ.get("SKIP_AUG") == "1"
    or MAX_FRAMES is not None
    or FRAME_IDX is not None
)

# HDR-refit meshes (sibling "<name>_hdr.glb") store sub-unity vertex colors
# that only resolve correctly under this exposure boost + ACES tone-map. The
# pipeline is a pure uint8 -> uint8 per-channel function, so we precompute a
# 256-entry LUT once and index the rendered RGB through it.
HDR_BOOST = 3.0
_HDR_TM_LUT = None


def _hdr_tm_lut(boost):
    s = np.arange(256, dtype=np.float32) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4) * boost
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    tm = np.clip((lin * (a * lin + b)) / (lin * (c * lin + d) + e), 0.0, 1.0)
    out = np.where(tm <= 0.0031308, 12.92 * tm, 1.055 * tm ** (1 / 2.4) - 0.055)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _apply_hdr_tm_rgba(rgba):
    global _HDR_TM_LUT
    if _HDR_TM_LUT is None:
        _HDR_TM_LUT = _hdr_tm_lut(HDR_BOOST)
    rgb = _HDR_TM_LUT[rgba[..., :3]]
    return np.concatenate([rgb, rgba[..., 3:4]], axis=-1)


def _sample_hsv(ranges, rng):
    return dict(
        hue_deg=float(rng.uniform(*ranges["hue_deg"])),
        sat=float(rng.uniform(*ranges["sat"])),
        val=float(rng.uniform(*ranges["val"])),
    )


def _apply_hsv_rgb(rgb, hsv):
    """HSV jitter on a final RGB composite (no alpha channel)."""
    return MainRenderPipeline._hsv_transform_rgb(
        np.ascontiguousarray(rgb), hsv["hue_deg"], hsv["sat"], hsv["val"]
    )


_GS_CALIB_ORIG = {}  # id(scene_manager) -> (T_dp_orig, s_dp_orig)


# Async PNG writer pool. cv2.imwrite releases the GIL during libpng encoding,
# so threads (cheap to create, no pickle cost) overlap encode with the next
# frame's render and saturate ~4 cores. Lazy-init so dry-run / argparse-only
# invocations don't spin up worker threads.
_PNG_WRITER = None
_PNG_WRITE_QUEUE_LIMIT = 48  # bounded backlog to cap pending-frame memory


def _get_png_writer():
    global _PNG_WRITER
    if _PNG_WRITER is None:
        _PNG_WRITER = ThreadPoolExecutor(max_workers=12, thread_name_prefix="png")
    return _PNG_WRITER


def _encode_and_write_png(out_path, rgb_image):
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr, IMG_PARAMS)


def _gs_calib_path(ply_path):
    """Per-splat calibration file: <ply_dir>/<ply_stem>/default.json."""
    stem = os.path.splitext(os.path.basename(str(ply_path)))[0]
    return os.path.join(os.path.dirname(str(ply_path)), stem, "default.json")


def _missing_calib_message(ply_path):
    calib_path = _gs_calib_path(ply_path)
    if asset_group(calib_path):
        return missing_asset_message(calib_path)
    stem = os.path.splitext(os.path.basename(str(ply_path)))[0]
    return (f"{calib_path} not found: the splat has no calibration to the sim world. "
            f"Create one with `python simulate_object.py --ply {stem}` (scripts/main, "
            "'GSplat Calibration' -> Save Calibration), or pass --allow-uncalibrated.")


def apply_gs_calibration(scene_manager, scene_name=None, allow_uncalibrated=False):
    """Apply the shared gsplat calibration to `scene_manager`, matching what
    gui_manager.build_gsplat_calibration does on Viser startup.

    Calibrations live in a per-ply directory (`<ply_dir>/<ply_stem>/`) so
    different splat maps can carry their own calibrations, but within a
    splat we always use the same `default.json` regardless of task scene
    — one calibration drives every scene, so a single nudge applies to
    every dataset built from this splat. The `scene_name` parameter is
    accepted for back-compat and logged but no longer changes the file
    chosen. Idempotent: the original `dataparser_transform` /
    `dataparser_scale` are cached on the first call so the calibration
    can be swapped without accumulating transforms across episodes.
    A missing `default.json` raises unless `allow_uncalibrated`.
    """
    key = id(scene_manager)
    if key not in _GS_CALIB_ORIG:
        _GS_CALIB_ORIG[key] = (
            scene_manager.T_gs_world.copy(),
            float(scene_manager.dataparser_scale),
        )
    T_dp_orig, s_dp_orig = _GS_CALIB_ORIG[key]

    ply_path = scene_manager.model_path
    calib_path = _gs_calib_path(ply_path)

    if not os.path.exists(calib_path):
        if not allow_uncalibrated:
            raise FileNotFoundError(_missing_calib_message(ply_path))
        # Uncalibrated render requested — restore originals so a previous
        # scene's calib doesn't leak into this one.
        scene_manager.T_gs_world = T_dp_orig.copy()
        scene_manager.dataparser_transform = (
            torch.tensor(T_dp_orig).float().to(scene_manager.device)
        )
        scene_manager.dataparser_scale = s_dp_orig
        print(f"[OfflineRender] WARNING: no GS calibration at {calib_path}; "
              "rendering the splat uncalibrated (--allow-uncalibrated)")
        return

    with open(calib_path, "r") as f:
        data = json.load(f)
    pos = np.asarray(data.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    rpy = data.get("rotation_rpy_deg", [0.0, 0.0, 0.0])
    s_calib = float(data.get("scale", 1.0))

    T_calib = np.eye(4, dtype=np.float32)
    T_calib[:3, :3] = R.from_euler("xyz", rpy, degrees=True).as_matrix().astype(np.float32)
    T_calib[:3, 3] = pos

    # Compose against originals so this is idempotent — repeated calls with
    # the same scene_name yield the same final transform.
    T_dp_new = (T_dp_orig @ np.linalg.inv(T_calib)).astype(np.float32)
    scene_manager.T_gs_world = T_dp_new
    scene_manager.dataparser_transform = torch.tensor(T_dp_new).float().to(scene_manager.device)
    scene_manager.dataparser_scale = s_dp_orig / s_calib
    print(f"[OfflineRender] Applied shared GS calibration from {calib_path}")


_WARNED_MISSING = set()  # downloadable meshes already warned about


def _resolve_mesh_path(entry):
    """Prefer HDR-refit (*_hdr.glb), then textured .glb, then configured file.

    HDR-refit meshes carry sub-unity vertex colors and require the HDR+TM
    post-process wired up in build_entity_renderers; the file suffix is the
    only marker the renderer keys off.
    """
    base_no_ext, _ = os.path.splitext(entry["mesh"])
    root = _OBJECTS_DIR
    hdr = os.path.join(root, base_no_ext + "_hdr.glb")
    if os.path.exists(hdr):
        return hdr
    if asset_group(hdr) and hdr not in _WARNED_MISSING:
        # A downloadable HDR refit exists for this mesh but isn't fetched.
        _WARNED_MISSING.add(hdr)
        print(f"[OfflineRender] WARNING: {missing_asset_message(hdr)}\n"
              f"  Rendering '{entry['name']}' with its un-refit mesh; colors will not match the paper.")
    for p in (os.path.join(root, base_no_ext + ".glb"), os.path.join(root, entry["mesh"])):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(missing_asset_message(os.path.join(root, base_no_ext + ".glb")))


def load_entity_mesh(entry, path=None):
    """Load mesh with texture support. GLB -> Y-up to Z-up correction."""
    path = path or _resolve_mesh_path(entry)
    if path.endswith('.obj'):
        loaded = trimesh.load(path)
        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
        png_path = path.replace('.obj', '.png')
        if os.path.exists(png_path) and getattr(mesh.visual, "kind", None) == 'texture':
            mesh.visual.material.image = Image.open(png_path)
    elif path.endswith('.glb'):
        loaded = trimesh.load(path)
        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
        correction = np.eye(4, dtype=np.float32)
        correction[:3, :3] = R.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
        mesh.apply_transform(correction)
    else:
        mesh = trimesh.load(path, force='mesh')
    return mesh


def build_entity_renderers(scene_cfg, viewport_w, viewport_h):
    """Return {name: (SimpleObjectRenderer, scale)} for every static + object.

    Entries with render_in_image=False are skipped (e.g. tables that are baked
    into the GS background already), matching simulate_object._add_static_to_viser.
    """
    renderers = {}
    for entry in list(scene_cfg.get("statics", [])) + list(scene_cfg.get("objects", [])):
        if not entry.get("render_in_image", True):
            continue
        name = entry["name"]
        resolved = _resolve_mesh_path(entry)  # raises if no mesh file exists
        mesh = load_entity_mesh(entry, resolved)
        rndr = SimpleObjectRenderer(viewport_width=viewport_w, viewport_height=viewport_h)
        rndr.set_mesh(mesh, smooth=False)
        if resolved.endswith("_hdr.glb"):
            _orig_render = rndr.render
            def _make_wrapped(orig):
                def wrapped(*args, **kwargs):
                    rgba, depth = orig(*args, **kwargs)
                    return _apply_hdr_tm_rgba(rgba), depth
                return wrapped
            rndr.render = _make_wrapped(_orig_render)
            print(f"[OfflineRender] HDR+TM enabled for '{name}'")
        renderers[name] = (rndr, float(entry.get("scale", 1.0)))
    return renderers


def composite_by_depth(img_gs, depth_gs, fg_layers):
    """Composite foreground layers onto GS background using per-pixel depth test."""
    H, W = img_gs.shape[:2]
    depth_gs = depth_gs.squeeze()

    final_img = img_gs.copy()
    final_depth = depth_gs.copy()

    if not fg_layers:
        return final_img, final_depth

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
        # Pyrender RGBA output is premultiplied at antialiased edges (MSAA resolves
        # covered + transparent samples → rgb scaled by coverage). Two paths:
        #   alpha == 255: flat copy (mathematically identical to "over", since
        #     255*fg + (255-255)*bg / 255 == fg). Skips the blend math entirely.
        #   alpha <  255: premultiplied "over" blend against the bg.
        # Keying off alpha (not cv2.erode of fg_vis as previously) avoids the
        # silhouette outline bug: where two fg layers overlap, the closer layer's
        # antialiased edge pixel has alpha < 255 and its neighbors are filled in
        # by the layer behind, so erosion misclassifies it as interior and the
        # premultiplied rgb gets copied directly, leaving a dark ring.
        opaque = alpha == 255
        final_img[opaque] = best_rgba[opaque, :3]

        boundary = fg_vis & ~opaque
        if boundary.any():
            sel_a = best_rgba[boundary, 3:4].astype(np.uint16)
            sel_fg = best_rgba[boundary, :3].astype(np.uint16)
            sel_bg = final_img[boundary].astype(np.uint16)
            final_img[boundary] = (sel_fg + (sel_bg * (255 - sel_a)) // 255).astype(np.uint8)

        final_depth[fg_vis] = best_depth[fg_vis]

    return final_img, final_depth


def T_yaw(yaw: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.from_euler("z", yaw).as_matrix().astype(np.float32)
    return T


def se2_about_anchor_T(anchor_xy, match_xy, yaw):
    anchor_xy = np.asarray(anchor_xy, dtype=np.float32)
    match_xy = np.asarray(match_xy, dtype=np.float32)
    T0 = np.eye(4, dtype=np.float32)
    T0[:2, 3] = -anchor_xy
    TR = T_yaw(float(yaw))
    T1 = np.eye(4, dtype=np.float32)
    T1[:2, 3] = match_xy
    return T1 @ TR @ T0


def apply_T_to_pose_xy_yaw(T, pos_xyz, quat_wxyz):
    pos_xyz = np.asarray(pos_xyz, dtype=np.float32)
    qw, qx, qy, qz = quat_wxyz
    Rw = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
    Tw = np.eye(4, dtype=np.float32)
    Tw[:3, :3] = Rw
    Tw[:3, 3] = pos_xyz
    Tw_new = T @ Tw
    pos_new = Tw_new[:3, 3]
    quat_xyzw = R.from_matrix(Tw_new[:3, :3]).as_quat()
    quat_new_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return pos_new, quat_new_wxyz


def _pose_to_T(pos_w, wxyz_w):
    if pos_w is None or wxyz_w is None:
        return None
    T = np.eye(4, dtype=np.float32)
    qw, qx, qy, qz = wxyz_w
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
    T[:3, 3] = np.asarray(pos_w, dtype=np.float32)
    return T


def render_episode(records, static_poses, scene_manager, robot_renderer,
                   entity_renderers, ep_out_dir, camera_metadata, joint_names,
                   T_aug=None, cameras=("d435",), enable_hsv_jitter=None,
                   lock_orient_names=None):
    """Render every frame.

    static_poses: {name: T_world_4x4} — fixed per episode (applied through T_aug).
    entity_renderers: {name: (renderer, scale)} — covers both statics and objects.
    enable_hsv_jitter: if True, sample one HSV triple from FINAL_HSV_RANGES at
        the start of this call and apply to every final RealSense-domain frame.
        None defaults to module-level ENABLE_HSV_JITTER.
    """
    if enable_hsv_jitter is None:
        enable_hsv_jitter = ENABLE_HSV_JITTER

    t_start = time.perf_counter()
    writer = _get_png_writer()
    write_futures = []  # bounded queue; oldest is drained when limit hit

    final_hsv = None
    if enable_hsv_jitter:
        rng = np.random.default_rng()
        final_hsv = _sample_hsv(FINAL_HSV_RANGES, rng)
        print(
            f"  HSV jitter (RealSense domain) — "
            f"hue={final_hsv['hue_deg']:+.1f}° "
            f"sat={final_hsv['sat']:.2f} val={final_hsv['val']:.2f}"
        )

    # Precompute static fg transforms so we don't redo per frame.
    static_T = {}
    for name, T in static_poses.items():
        if T is None or name not in entity_renderers:
            continue
        static_T[name] = (T_aug @ T).astype(np.float32) if T_aug is not None else T

    for frame_idx, record in enumerate(records):
        if FRAME_IDX is not None and frame_idx != FRAME_IDX:
            continue
        if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
            break
        rob = record["robot"]

        T_base = np.eye(4, dtype=np.float32)
        qw, qx, qy, qz = rob["base_wxyz_w"]
        T_base[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        T_base[:3, 3] = np.asarray(rob["base_pos_w"], dtype=np.float32)

        # Per-frame dynamic object transforms (post-augmentation).
        object_T = {}
        frame_objects = record.get("objects") or {}
        for name, data in frame_objects.items():
            if name not in entity_renderers:
                continue
            T_w = _pose_to_T(data.get("pos_w"), data.get("wxyz_w"))
            if T_w is None:
                continue
            if lock_orient_names and name in lock_orient_names:
                T_w[:3, :3] = np.eye(3, dtype=T_w.dtype)
            object_T[name] = (T_aug @ T_w).astype(np.float32) if T_aug is not None else T_w

        joints_dict = dict(zip(joint_names, rob["joints"]))

        W, H = 640, 480
        fov = float(camera_metadata.get("d435", {}).get("fov", 0.74))

        for cam_name, cam_data in record["cameras"].items():
            if cam_name not in cameras:
                continue
            pos_w = cam_data["pos_w"]
            qw_c, qx_c, qy_c, qz_c = cam_data["wxyz_w"]

            T_cam = np.eye(4, dtype=np.float32)
            T_cam[:3, :3] = R.from_quat([qx_c, qy_c, qz_c, qw_c]).as_matrix()
            T_cam[:3, 3] = pos_w

            # GS background + robot foreground. The robot/table/object are
            # essentially always visible from the d435 view, so the np.any
            # alpha prechecks were paying for a full-frame scan to skip nothing.
            K = D435_REAL_K if cam_name == "d435" else None
            img_gs, depth_gs = scene_manager.render_camera(T_cam, W, H, fov, K=K)
            img_rob, depth_rob = robot_renderer.render(joints_dict, T_base, T_cam, W, H, fov, K=K)
            fg_layers = [(img_rob, depth_rob)]

            # Statics (constant per episode) then dynamic objects (per frame).
            for name, T_world in static_T.items():
                rndr, scale = entity_renderers[name]
                img_e, depth_e = rndr.render(T_cam, T_world, scale, W, H, fov, K=K)
                fg_layers.append((img_e, depth_e))

            for name, T_world in object_T.items():
                rndr, scale = entity_renderers[name]
                img_e, depth_e = rndr.render(T_cam, T_world, scale, W, H, fov, K=K)
                fg_layers.append((img_e, depth_e))

            final_img, _ = composite_by_depth(img_gs, depth_gs, fg_layers)
            # iPhone-domain composite → RealSense-domain via fixed CCM.
            if APPLY_IPHONE_TO_REALSENSE:
                final_img = calibrate_iphone_to_realsense(final_img)
            # Single HSV jitter pass in the RealSense domain (domain rand.)
            if final_hsv is not None:
                final_img = _apply_hsv_rgb(final_img, final_hsv)

            out_path = os.path.join(ep_out_dir, f"{frame_idx:05d}_{cam_name}.{IMG_EXT}")
            # final_img is a fresh buffer from composite_by_depth's
            # img_gs.copy(), so the worker can hold the reference without
            # racing the next iteration's rebind.
            write_futures.append(writer.submit(_encode_and_write_png, out_path, final_img))
            if len(write_futures) >= _PNG_WRITE_QUEUE_LIMIT:
                write_futures.pop(0).result()

        print(f"  Rendered Frame {frame_idx:05d}/{len(records)}", end="\r")
    print()
    # Drain remaining writes before reporting timing; otherwise the timing
    # window ends before disk I/O completes and overstates throughput.
    for f in write_futures:
        f.result()
    t_elapsed = time.perf_counter() - t_start
    n = len(records)
    fps = n / t_elapsed if t_elapsed > 0 else 0.0
    print(f"  [Timing] {n} frames in {t_elapsed:.2f}s -> {fps:.2f} fps "
          f"({1000.0 * t_elapsed / max(n, 1):.1f} ms/frame)")


def _parse_scenario(scenario, override_scene=None):
    """Extract (scene_cfg, scene_name, static_poses, camera_metadata, joint_names).

    Supports both the new format (statics list + scene_name) and the legacy
    format (table_pos_w + obj_pos_w) by mapping legacy fields onto the first
    static / first object in the resolved scene.
    """
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

    # Legacy fallback: single "table" pose -> first static.
    if not static_poses and scenario.get("table_pos_w") and scene_cfg.get("statics"):
        T = _pose_to_T(scenario.get("table_pos_w"), scenario.get("table_wxyz_w"))
        if T is not None:
            static_poses[scene_cfg["statics"][0]["name"]] = T

    camera_metadata = scenario.get("camera_metadata", {})
    joint_names = scenario.get("joint_names", [])
    return scene_cfg, scene_name, static_poses, camera_metadata, joint_names


def _upgrade_legacy_frames(records, scene_cfg):
    """If frames carry the old single-"object" field, remap to frame["objects"]
    keyed by the first object's name so the new renderer can consume them.
    """
    if not records or not scene_cfg.get("objects"):
        return records
    primary_name = scene_cfg["objects"][0]["name"]
    upgraded = []
    for rec in records:
        if rec.get("objects") or "object" not in rec:
            upgraded.append(rec)
            continue
        rec2 = dict(rec)
        obj = rec["object"] or {}
        if obj.get("pos_w") is not None:
            rec2["objects"] = {primary_name: {
                "pos_w": obj.get("pos_w"),
                "wxyz_w": obj.get("wxyz_w"),
            }}
        upgraded.append(rec2)
    return upgraded


def render_episode_files(robot_files, data_dir, ply_path,
                          scene_override=None, cameras=("d435",),
                          enable_hsv_jitter=True, worker_tag="",
                          allow_uncalibrated=False):
    """Render the given episode JSONs end-to-end.

    Builds one scene_manager + robot_renderer per call (so a multi-process
    wrapper can call this in each worker to give every process its own GPU
    state). The renderer_cache is local to this call.
    """
    if not robot_files:
        return
    output_dir = os.path.join(data_dir, OUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    base_W, base_H = 640, 480

    tag = f"[{worker_tag}] " if worker_tag else ""
    print(f"{tag}Initializing Offline Renderers (ply={ply_path})...")
    scene_manager = VisualizerSceneManager(ply_path)
    robot_renderer = RobotMeshRenderer(yourdfpy.URDF.load(URDF_PATH), URDF_PATH)

    renderer_cache = {}
    current_calib_scene = object()

    for rob_file in robot_files:
        ep_str = os.path.basename(rob_file).replace("episode_", "").replace(".json", "")
        ep_out_dir = os.path.join(output_dir, f"episode_{ep_str}")
        os.makedirs(ep_out_dir, exist_ok=True)
        print(f"\n{tag}--- Processing Episode {ep_str} ---")

        scenario_file = os.path.join(data_dir, f"episode_{ep_str}_scenario.json")
        if not os.path.exists(scenario_file):
            print(f"  Missing {scenario_file}; skipping")
            continue
        with open(scenario_file, "r") as f:
            scenario = json.load(f)

        scene_cfg, scene_name, static_poses, camera_metadata, joint_names = \
            _parse_scenario(scenario, override_scene=scene_override)

        if scene_name != current_calib_scene:
            apply_gs_calibration(scene_manager, scene_name=scene_name,
                                 allow_uncalibrated=allow_uncalibrated)
            current_calib_scene = scene_name

        if scene_name not in renderer_cache:
            print(f"  Building entity renderers for scene '{scene_name}'...")
            renderer_cache[scene_name] = build_entity_renderers(scene_cfg, base_W, base_H)
        entity_renderers = renderer_cache[scene_name]
        lock_orient_names = {
            e["name"] for e in scene_cfg.get("objects", [])
            if e.get("lock_world_orientation")
        }

        with open(rob_file, "r") as f:
            records = json.load(f)
        records = _upgrade_legacy_frames(records, scene_cfg)

        orig_dir = os.path.join(ep_out_dir, "00")
        os.makedirs(orig_dir, exist_ok=True)
        print(f"{tag}Rendering original...")
        render_episode(
            records, static_poses, scene_manager, robot_renderer, entity_renderers,
            orig_dir, camera_metadata, joint_names, T_aug=None, cameras=cameras,
            enable_hsv_jitter=enable_hsv_jitter,
            lock_orient_names=lock_orient_names,
        )

        if SKIP_AUG:
            print(f"{tag}SKIP_AUG/MAX_FRAMES/FRAME_IDX set; skipping augmented trajectory renders.")
            continue

        aug_tf_file = os.path.join(data_dir, f"episode_{ep_str}_object_aug_transforms.json")
        if not os.path.exists(aug_tf_file):
            print(f"{tag}No augmented transforms found; skipping augmentation renders.")
            continue

        with open(aug_tf_file, "r") as f:
            aug_data = json.load(f)

        anchor_xy = np.array(aug_data["mask_center_world_xy"], dtype=np.float32)
        transforms = aug_data["transforms"]

        print(f"{tag}Rendering {len(transforms)} augmentations...")

        for k, tf in enumerate(transforms):
            match_xy = np.array([tf["x"], tf["y"]], dtype=np.float32)
            yaw = float(tf["yaw"])
            T_aug = se2_about_anchor_T(anchor_xy, match_xy, yaw)

            records_aug = []
            for rec in records:
                rec2 = json.loads(json.dumps(rec))
                rob = rec2["robot"]
                pos_new, quat_new = apply_T_to_pose_xy_yaw(T_aug, rob["base_pos_w"], rob["base_wxyz_w"])
                rob["base_pos_w"] = pos_new.tolist()
                rob["base_wxyz_w"] = quat_new.tolist()
                for _, cam_data in rec2["cameras"].items():
                    pos_new, quat_new = apply_T_to_pose_xy_yaw(T_aug, cam_data["pos_w"], cam_data["wxyz_w"])
                    cam_data["pos_w"] = pos_new.tolist()
                    cam_data["wxyz_w"] = quat_new.tolist()
                records_aug.append(rec2)

            aug_dir = os.path.join(ep_out_dir, f"{k:02d}")
            os.makedirs(aug_dir, exist_ok=True)
            print(f"  {tag}Rendering augmented trajectory {k:02d}...")
            render_episode(
                records_aug, static_poses, scene_manager, robot_renderer, entity_renderers,
                aug_dir, camera_metadata, joint_names, T_aug=T_aug, cameras=cameras,
                enable_hsv_jitter=enable_hsv_jitter,
                lock_orient_names=lock_orient_names,
            )

        print(f"{tag}Done with episode augmentations.")

    print(f"\n{tag}Offline rendering complete!")


def _resolve_ply_path(ply):
    """Bare name -> <repo>/data/gsplat/<name>.ply; anything else is a path."""
    if ply and "/" not in ply and os.sep not in ply:
        name = ply if ply.endswith(".ply") else ply + ".ply"
        return str(_REPO_ROOT / "data" / "gsplat" / name)
    return ply


def check_scene_assets(scene_name):
    """Fail fast (before any GPU work) if a --scene is unknown or misses a mesh."""
    if scene_name is None:
        return
    if scene_name not in SCENES:
        raise SystemExit(f"Unknown --scene '{scene_name}'. Known scenes: {', '.join(sorted(SCENES))}")
    cfg = SCENES[scene_name]
    for entry in list(cfg.get("statics", [])) + list(cfg.get("objects", [])):
        if entry.get("render_in_image", True):
            try:
                _resolve_mesh_path(entry)
            except FileNotFoundError as e:
                raise SystemExit(str(e))


def resolve_splat(ply=None, scene=None, allow_uncalibrated=False):
    """Pick the splat for a run and check its files, and the core assets,
    before any GPU work.

    An explicit --ply wins; otherwise the --scene variant's "gsplat" (the
    blue-table capture for the *_blue variants), else the default map.
    """
    require_core_assets()
    if scene is not None and scene not in SCENES:
        raise SystemExit(f"Unknown --scene '{scene}'. Known scenes: {', '.join(sorted(SCENES))}")
    if ply is None:
        ply = (SCENES[scene].get("gsplat") if scene else None) or DEFAULT_MAP
    ply_path = _resolve_ply_path(ply)
    if not os.path.exists(ply_path):
        hint = " (or render over the flightroom splat from the core assets: --ply flightroom)" \
            if ply == DEFAULT_MAP else ""
        raise SystemExit(missing_asset_message(ply_path) + hint)
    if not allow_uncalibrated and not os.path.exists(_gs_calib_path(ply_path)):
        raise SystemExit(_missing_calib_message(ply_path))
    return ply_path


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
        print(f"[OfflineRender] --episode-from {episode_from}: {before} -> {len(robot_files)} files")
    return robot_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATASET_DIR,
                        help="Episode dir (reads episode_*.json, writes final_renders/).")
    parser.add_argument("--episode", type=str, default=None,
                        help="Render a single episode, e.g. '0000'")
    parser.add_argument("--episode-from", type=int, default=None,
                        help="Skip episodes below this index (resume after partial run).")
    parser.add_argument("--mid360", action="store_true",
                        help="Also render mid360 camera (default: d435 only)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Render with this scene instead of the recorded one, e.g. an "
                             "appearance variant such as apple_to_box_turn (scene_config.py).")
    parser.add_argument("--hsv-jitter", action="store_true",
                        help="Enable per-episode HSV jitter (off by default; "
                             "stacked on top of HDR mesh + iphone2realsense).")
    parser.add_argument("--no-hsv-jitter", action="store_true",
                        help="(Deprecated; default is off.) Force HSV jitter off.")
    parser.add_argument("--ply", type=str, default=None,
                        help="Path to .ply gsplat; a bare name expands to data/gsplat/<name>.ply. "
                             f"Default: the --scene variant's splat, else {DEFAULT_MAP}.")
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="Render a splat that has no <ply_dir>/<name>/default.json "
                             "(it will not line up with the sim world).")
    args = parser.parse_args()

    ply_path = resolve_splat(args.ply, args.scene, args.allow_uncalibrated)
    check_scene_assets(args.scene)
    enable_hsv_jitter = (ENABLE_HSV_JITTER or args.hsv_jitter) and not args.no_hsv_jitter
    cameras = ("d435", "mid360") if args.mid360 else ("d435",)

    robot_files = _enumerate_episode_files(
        args.data_dir, episode=args.episode, episode_from=args.episode_from,
    )
    if not robot_files:
        raise SystemExit(f"Error: No episode data found in {args.data_dir}")

    render_episode_files(
        robot_files, args.data_dir, ply_path,
        scene_override=args.scene, cameras=cameras,
        enable_hsv_jitter=enable_hsv_jitter,
        allow_uncalibrated=args.allow_uncalibrated,
    )


if __name__ == "__main__":
    main()
