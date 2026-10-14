"""iPhone -> RealSense color calibration test renderer.

Renders the first N episodes of a collected dataset using the iPhone-captured
gsplat (default: the renderer's, `wood.ply`) at RealSense D435 intrinsics. Produces
one side-by-side MP4 per episode:

    [ raw composite | iPhone->RealSense calibrated composite ]

In both halves the mesh foreground (robot + statics + objects) is composited
on top of the gsplat by depth. The only difference between the halves is that
the right half applies `iphone_to_realsense.calibrate_iphone_to_realsense` to
the gsplat layer *before* compositing. HSV jitter is disabled on both halves.

Purpose: visually compare against an actual RealSense capture of the same
room to see whether the CCM closes the iPhone-gsplat / RealSense-camera
color gap.

Usage:
    python scripts/data/test_iphone_calib_render.py \\
        --data-dir <episode-dir> --episodes 5 --fps 30
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYRENDER_OFFSCREEN", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch  # noqa: F401  (loaded by VisualizerSceneManager downstream)
import yourdfpy
from scipy.spatial.transform import Rotation as R

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "main"))
sys.path.insert(0, str(_REPO / "data"))

from utils.gs_rendering_utils import VisualizerSceneManager  # noqa: E402
from renderer.robot_renderer import RobotMeshRenderer  # noqa: E402
from renderer.iphone_to_realsense import calibrate_iphone_to_realsense  # noqa: E402

import offline_renderer as off  # noqa: E402


URDF_PATH = str(_REPO.parent / "robots/g1/urdf/g1_29dof_with_hand.urdf")
W, H = 640, 480


def _render_composite(img_gs, depth_gs, robot_renderer, entity_renderers,
                      joints_dict, T_base, T_cam, fov,
                      static_T, object_T):
    """Layer mesh foreground onto an already-rendered gsplat background by depth."""
    img_rob, depth_rob = robot_renderer.render(
        joints_dict, T_base, T_cam, W, H, fov
    )
    fg_layers = [(img_rob, depth_rob)]
    for name, T_world in static_T.items():
        rndr, scale = entity_renderers[name]
        img_e, depth_e = rndr.render(T_cam, T_world, scale, W, H, fov)
        fg_layers.append((img_e, depth_e))
    for name, T_world in object_T.items():
        rndr, scale = entity_renderers[name]
        img_e, depth_e = rndr.render(T_cam, T_world, scale, W, H, fov)
        fg_layers.append((img_e, depth_e))
    final, _ = off.composite_by_depth(img_gs, depth_gs, fg_layers)
    return final


def _render_episode_pair(records, static_poses, scene_manager, robot_renderer,
                         entity_renderers, joint_names, camera_metadata,
                         video_path, fps):
    fov = float(camera_metadata.get("d435", {}).get("fov", 0.74))
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (W * 2, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {video_path}")

    static_T = {n: T for n, T in static_poses.items() if n in entity_renderers}

    t0 = time.perf_counter()
    n = len(records)
    for idx, record in enumerate(records):
        rob = record["robot"]
        T_base = np.eye(4, dtype=np.float32)
        qw, qx, qy, qz = rob["base_wxyz_w"]
        T_base[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        T_base[:3, 3] = np.asarray(rob["base_pos_w"], dtype=np.float32)
        joints_dict = dict(zip(joint_names, rob["joints"]))

        object_T = {}
        for name, data in (record.get("objects") or {}).items():
            if name not in entity_renderers:
                continue
            T_w = off._pose_to_T(data.get("pos_w"), data.get("wxyz_w"))
            if T_w is not None:
                object_T[name] = T_w

        cam_data = record["cameras"].get("d435")
        if cam_data is None:
            continue
        qw_c, qx_c, qy_c, qz_c = cam_data["wxyz_w"]
        T_cam = np.eye(4, dtype=np.float32)
        T_cam[:3, :3] = R.from_quat([qx_c, qy_c, qz_c, qw_c]).as_matrix().astype(np.float32)
        T_cam[:3, 3] = np.asarray(cam_data["pos_w"], dtype=np.float32)

        img_gs, depth_gs = scene_manager.render_camera(T_cam, W, H, fov)
        img_gs = np.ascontiguousarray(img_gs)
        depth_gs = np.ascontiguousarray(depth_gs)
        img_gs_calib = calibrate_iphone_to_realsense(img_gs)

        # depth_gs is shared; mesh foreground is identical in both halves.
        # Compositing twice is cheap (~ms) compared to gsplat render (~50ms).
        raw_frame = _render_composite(
            img_gs, depth_gs, robot_renderer, entity_renderers,
            joints_dict, T_base, T_cam, fov, static_T, object_T,
        )
        calib_frame = _render_composite(
            img_gs_calib, depth_gs, robot_renderer, entity_renderers,
            joints_dict, T_base, T_cam, fov, static_T, object_T,
        )

        side = np.concatenate([raw_frame, calib_frame], axis=1)
        writer.write(cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
        if (idx + 1) % 25 == 0 or idx + 1 == n:
            print(f"    frame {idx + 1}/{n}", end="\r", flush=True)

    writer.release()
    dt = time.perf_counter() - t0
    print(f"\n    wrote {video_path}  ({n} frames, {dt:.1f}s, {n / max(dt, 1e-6):.1f} fps)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ply", type=str, default=None,
                        help="ply path or bare name under data/gsplat/ "
                             f"(default: {off.DEFAULT_MAP})")
    parser.add_argument("--episodes", type=int, default=5,
                        help="number of episodes from the start (default: 5)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=str, default=None,
                        help="output dir (default: <data-dir>/iphone_calib_test)")
    args = parser.parse_args()

    data_dir = args.data_dir
    ply_path = off.resolve_splat(args.ply)  # exits if the splat or its calibration is missing
    out_dir = args.out or os.path.join(data_dir, "iphone_calib_test")
    os.makedirs(out_dir, exist_ok=True)

    robot_files = off._enumerate_episode_files(data_dir)
    if not robot_files:
        sys.exit(f"no episodes found in {data_dir}")
    robot_files = robot_files[: args.episodes]
    print(f"Rendering {len(robot_files)} episode(s) from {data_dir}")
    print(f"  ply = {ply_path}")
    print(f"  out = {out_dir}")

    print("Loading scene manager + robot renderer...")
    scene_manager = VisualizerSceneManager(ply_path)
    robot_renderer = RobotMeshRenderer(yourdfpy.URDF.load(URDF_PATH), URDF_PATH)

    renderer_cache = {}
    current_calib_scene = object()

    for rob_file in robot_files:
        ep_str = os.path.basename(rob_file).replace("episode_", "").replace(".json", "")
        scenario_file = os.path.join(data_dir, f"episode_{ep_str}_scenario.json")
        if not os.path.exists(scenario_file):
            print(f"  [skip] missing {scenario_file}")
            continue
        with open(scenario_file) as f:
            scenario = json.load(f)

        scene_cfg, scene_name, static_poses, camera_metadata, joint_names = \
            off._parse_scenario(scenario)

        if scene_name != current_calib_scene:
            off.apply_gs_calibration(scene_manager, scene_name=scene_name)
            current_calib_scene = scene_name

        if scene_name not in renderer_cache:
            renderer_cache[scene_name] = off.build_entity_renderers(scene_cfg, W, H)
        entity_renderers = renderer_cache[scene_name]

        with open(rob_file) as f:
            records = json.load(f)
        records = off._upgrade_legacy_frames(records, scene_cfg)

        video_path = os.path.join(out_dir, f"episode_{ep_str}_compare.mp4")
        print(f"\n--- episode {ep_str} ({len(records)} frames) -> {video_path}")
        _render_episode_pair(
            records, static_poses, scene_manager, robot_renderer,
            entity_renderers, joint_names, camera_metadata,
            video_path, args.fps,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
