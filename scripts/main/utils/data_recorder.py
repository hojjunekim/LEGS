import os, json, time
import numpy as np
from scipy.spatial.transform import Rotation as R


class DataRecorder:
    """
    Lightweight recorder for real-time simulator.
    Saves per-episode scenario JSON + per-frame pose JSON.
    No image rendering or LeRobot dataset creation — those happen offline.

    Saved files per episode:
        episode_XXXX_scenario.json  — one-time metadata (initial pose, goal, path, etc.)
        episode_XXXX.json           — list of per-frame records
    """

    def __init__(self, output_dir=None, fps=30,
                 language_instruction="grab a bottle", record_delay_sec=1.0):
        self.output_dir = output_dir
        self.fps = fps
        self.language_instruction = language_instruction
        self.record_delay_sec = float(record_delay_sec)

        self.episode_idx = -1
        self.episode_start_time = None
        self.joint_names = []
        self.frames = []

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[Recorder] Initialized. fps={fps}, output={output_dir}")
        else:
            print(f"[Recorder] Initialized. fps={fps}, output=<deferred — set on 🔴 Auto Record>")

    def start_new_episode(self, robot_pos_w, robot_wxyz_w,
                          joint_names, camera_metadata, planned_path_w=None,
                          blocked=0, scene_name=None,
                          statics=None, objects_meta=None,
                          language_instruction=None):
        """Record per-episode scenario.

        `statics`: list of {name, mesh, pos_w, wxyz_w, scale} — captured once;
                   statics are welded so their pose is invariant per episode.
        `objects_meta`: list of {name, mesh, scale} — metadata only. Per-frame
                   dynamic poses go into `record_robot_state(object_poses=...)`.
        """
        self._save_current_episode()

        self.episode_idx += 1
        self.episode_start_time = time.time()
        self.joint_names = list(joint_names)
        self.frames = []

        def _to_list(v):
            return v.tolist() if hasattr(v, "tolist") else list(v)

        statics_out = []
        for s in (statics or []):
            statics_out.append({
                "name": s["name"],
                "mesh": s.get("mesh"),
                "pos_w": _to_list(s["pos_w"]),
                "wxyz_w": _to_list(s["wxyz_w"]),
                "scale": float(s.get("scale", 1.0)),
            })

        objects_out = []
        for o in (objects_meta or []):
            objects_out.append({
                "name": o["name"],
                "mesh": o.get("mesh"),
                "scale": float(o.get("scale", 1.0)),
            })

        scenario_file = os.path.join(self.output_dir, f"episode_{self.episode_idx:04d}_scenario.json")
        os.makedirs(self.output_dir, exist_ok=True)
        instruction = language_instruction if language_instruction else self.language_instruction
        with open(scenario_file, 'w') as f:
            json.dump({
                "episode_idx": self.episode_idx,
                "blocked": int(bool(blocked)),
                "language_instruction": instruction,
                "scene_name": scene_name,
                "start_pos_w": robot_pos_w.tolist(),
                "start_wxyz_w": robot_wxyz_w.tolist(),
                "statics": statics_out,
                "objects": objects_out,
                "joint_names": self.joint_names,
                "camera_metadata": {
                    "d435": {"W": 224, "H": 224, "fov": camera_metadata.get("d435", {}).get("fov", 0.74)},
                },
                "planned_path_w": planned_path_w.tolist() if planned_path_w is not None else None,
            }, f, indent=4)

        print(f"[Recorder] Started Episode {self.episode_idx}. Scenario metadata saved.")

    def record_robot_state(self, robot_pos, robot_quat_wxyz, joints_dict,
                           T_d435_world, action_cmd, status=0,
                           object_poses=None,
                           L_mat=None, R_mat=None, H_mat=None,
                           left_gripper=0.0, right_gripper=0.0):
        """Record a per-frame observation.

        `object_poses`: dict {name: (pos_w, wxyz_w)} for dynamic (freejoint)
            objects. Serialized into frame["objects"][name] = {pos_w, wxyz_w}.
        `H_mat`: 4x4 head/neck pose in world frame. Used offline to fill
            SONIC's `teleop.vr_3pt_position/orientation` neck slot.
        """
        if self.episode_idx < 0:
            return

        if self.episode_start_time is not None:
            if (time.time() - self.episode_start_time) < self.record_delay_sec:
                return

        # Extract camera poses from 4x4 transforms
        def t44_to_pos_wxyz(T):
            pos = T[:3, 3].tolist()
            q_xyzw = R.from_matrix(T[:3, :3]).as_quat()
            wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
            return pos, wxyz

        d435_pos, d435_wxyz = t44_to_pos_wxyz(T_d435_world)

        # CLOCK_REALTIME on Linux — must match the deploy's state_logger
        # `time_realtime_ms` field so motion_token CSV joins by nearest neighbor.
        time_realtime_ms = time.clock_gettime(time.CLOCK_REALTIME) * 1000.0

        frame = {
            "time_realtime_ms": time_realtime_ms,
            "robot": {
                "base_pos_w": np.asarray(robot_pos, dtype=np.float32).tolist(),
                "base_wxyz_w": np.asarray(robot_quat_wxyz, dtype=np.float32).tolist(),
                "joints": [float(joints_dict[name]) for name in self.joint_names],
            },
            "cameras": {
                "d435": {"pos_w": d435_pos, "wxyz_w": d435_wxyz},
            },
            "action_cmd": {
                "vx": float(action_cmd.get("vx", 0.0)),
                "vy": float(action_cmd.get("vy", 0.0)),
                "vyaw": float(action_cmd.get("vyaw", 0.0)),
                "height": float(action_cmd.get("height", 0.78)),
            },
            "L_mat": np.asarray(L_mat, dtype=np.float32).tolist() if L_mat is not None else None,
            "R_mat": np.asarray(R_mat, dtype=np.float32).tolist() if R_mat is not None else None,
            "H_mat": np.asarray(H_mat, dtype=np.float32).tolist() if H_mat is not None else None,
            "left_gripper": float(left_gripper),
            "right_gripper": float(right_gripper),
            "status": int(status),
        }

        if object_poses:
            objs = {}
            for name, pose in object_poses.items():
                if pose is None:
                    continue
                pos_w, wxyz_w = pose
                objs[name] = {
                    "pos_w": np.asarray(pos_w, dtype=np.float32).tolist(),
                    "wxyz_w": np.asarray(wxyz_w, dtype=np.float32).tolist() if wxyz_w is not None else None,
                }
            if objs:
                frame["objects"] = objs

        self.frames.append(frame)

    def _save_current_episode(self):
        if self.episode_idx >= 0 and len(self.frames) > 0:
            ep_file = os.path.join(self.output_dir, f"episode_{self.episode_idx:04d}.json")
            with open(ep_file, 'w') as f:
                json.dump(self.frames, f, indent=2)
            print(f"[Recorder] Saved episode {self.episode_idx} ({len(self.frames)} frames).")
            self.frames = []

    def discard_current_episode(self):
        """Drop the in-progress episode: clear frames, remove the scenario
        file, and roll the episode counter back so the next ``start_new_episode``
        reuses the same idx."""
        if self.episode_idx < 0:
            return
        self.frames = []
        scenario_file = os.path.join(
            self.output_dir, f"episode_{self.episode_idx:04d}_scenario.json"
        )
        if os.path.exists(scenario_file):
            os.remove(scenario_file)
        print(f"[Recorder] Discarded episode {self.episode_idx}.")
        self.episode_idx -= 1

    def finalize(self):
        self._save_current_episode()
        print("[Recorder] Recording finalized.")

    def save_map_context(self, planner):
        # output_dir is None until the first 🔴 Auto Record click (deferred
        # init); skip the write and let the auto-record handler re-call us
        # once the timestamped directory exists.
        if self.output_dir is None:
            return
        map_file = os.path.join(self.output_dir, "map_context.npz")
        np.savez(
            map_file,
            grid_raw=planner.grid_raw.astype(np.uint8),
            grid_origin=planner.grid_origin.astype(np.float32),
            resolution=np.array([planner.resolution], dtype=np.float32),
            grid_shape=np.array(planner.grid_shape, dtype=np.int32),
        )
        print(f"[Recorder] Saved map context to {map_file}")
