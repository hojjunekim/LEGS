import math
import os
import random
import time
import numpy as np
from scene_config import resolve_randomization

class AppEventHandler:
    def __init__(self, app):
        self.app = app

    def process_events(self):
        """Processes all single-fire triggers from the GUI."""
        if self.app.gui.save_image_trigger != self.app.is_recording:
            self._handle_recording_toggle()
            
        if self.app.gui.walk_trigger:
            self._handle_walk()

        if self.app.gui.grab_trigger:
            self._handle_grab()

        if self.app.gui.auto_record_trigger:
            self._handle_auto_record()

        if self.app.gui.stop_trigger:
            self._handle_stop()
            
        if self.app.gui.reset_sim_trigger:
            self._handle_reset_sim()

        if self.app.gui.task_next_trigger:
            self._handle_task_next()

        if self.app.gui.task_run_trigger:
            self._handle_task_run()

        if self.app.gui.task_reset_trigger:
            self._handle_task_reset()

        # Continuous poll (not a single-fire trigger): auto-stop a manual
        # single-episode recording once its task plan completes.
        self._update_single_record()

    def _ensure_output_dir(self):
        """Lazily create a timestamped out_dir on the first record click of this
        session if the user didn't pass --out_dir at launch. Shared by the
        single-episode 🔴 Toggle Recording button and multi-episode 🔴 Auto
        Record so episode_idx stays monotonic within a sim run. No-op once set."""
        if self.app.recorder.output_dir is not None:
            return
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        task_name = getattr(self.app, "_task_name", None)
        sub = f"{task_name}_{stamp}" if task_name else stamp
        new_dir = os.path.join(self.app._collected_root, sub)
        os.makedirs(new_dir, exist_ok=True)
        self.app.recorder.output_dir = new_dir
        print(f"[Recorder] Output dir set: {new_dir}")
        # setup_scene's first save_map_context no-op'd while output_dir was
        # None; emit it now so the map context lands in this run.
        self.app.recorder.save_map_context(self.app.planner)

    def _update_single_record(self):
        """Stop+save a manual single-episode recording when the task plan
        succeeds. Mirrors Auto Record's success signal (task manager idle +
        done + not failed) but only for manual recordings — Auto Record owns
        its own stop (skipped here via the _auto_rec["active"] guard). The
        _single_rec_task_ran latch ensures we only stop after the plan actually
        ran this episode, so toggling Record on after an already-finished (or
        empty) plan doesn't insta-save."""
        if not self.app.is_recording or self.app._auto_rec.get("active"):
            return
        tm = getattr(self.app, "task_manager", None)
        if tm is None:
            return  # scene has no task_plan → Toggle is a pure manual on/off
        if tm.mode != "idle":
            self.app._single_rec_task_ran = True
        elif self.app._single_rec_task_ran and tm.is_done() and not tm.failed():
            print("[Recorder] Task plan complete — auto-stopping single-episode recording.")
            # Saved next tick via _handle_recording_toggle's else branch.
            self.app.gui.save_image_trigger = False
            self.app._single_rec_task_ran = False

    def _handle_recording_toggle(self):
        # On the first click of a manual recording, lazily create an out_dir
        # (same as Auto Record) so a single episode saves without --out_dir or
        # a prior Auto Record click. Arm the task-success latch fresh so
        # auto-stop only fires after the plan actually runs this episode.
        if self.app.gui.save_image_trigger:
            self._ensure_output_dir()
            self.app._single_rec_task_ran = False

        self.app.is_recording = self.app.gui.save_image_trigger
        if self.app.is_recording:
            # Reset sequence states so status starts fresh
            self.app.controller.reached_goal = False
            self.app.controller.at_object_location = False
            self.app.upper_controller.reset()
            self.app.arrival_time = None

            self.app.grab_time = None
            self.app._approach_settle_time = None

            state = self.app.ros_node.get_latest_state()
            if state:
                scene_cfg = self.app.scene_cfg
                statics = []
                for s in scene_cfg.get("statics", []):
                    pose = self.app.ros_node.get_pose(s["name"])
                    if pose is None:
                        pos_w = np.asarray(s["pos"], dtype=np.float32)
                        wxyz_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        pos_w, wxyz_w = pose
                    statics.append({
                        "name": s["name"], "mesh": s["mesh"], "scale": s.get("scale", 1.0),
                        "pos_w": pos_w, "wxyz_w": wxyz_w,
                    })
                objects_meta = [
                    {"name": o["name"], "mesh": o["mesh"], "scale": o.get("scale", 1.0)}
                    for o in scene_cfg.get("objects", [])
                ]

                self.app.recorder.start_new_episode(
                    robot_pos_w=state[0], robot_wxyz_w=state[1],
                    joint_names=list(state[2].keys()),
                    camera_metadata=self.app.get_camera_metadata(),
                    planned_path_w=self.app.path,
                    scene_name=getattr(self.app.args, "scene", None),
                    statics=statics,
                    objects_meta=objects_meta,
                    # The prompt sent to the policy, if one is connected
                    language_instruction=getattr(self.app, "policy_prompt", None) or scene_cfg.get("prompt"),
                )
        else:
            self.app.recorder._save_current_episode()

    def _handle_walk(self):
        state = self.app.ros_node.get_latest_state()
        if state is not None:
            robot_pos = state[0]
            target_pos = np.array(self.app.gui.obj_pose)
            self.app.path = self.app.planner.plan_path(robot_pos, target_pos)
            
            if self.app.path is not None:
                dx, dy = target_pos[0] - robot_pos[0], target_pos[1] - robot_pos[1]
                self.app.controller.set_path(
                    self.app.path, 
                    end_action_name=self.app.current_action_name, 
                    direct_heading=np.arctan2(dy, dx)
                )
                self.app.server.scene.add_spline_catmull_rom(
                    "/planned_path", positions=self.app.path, color=(25, 220, 50), line_width=4.0
                )
                
        # Reset the upper body and the arrival timer.
        self.app.upper_controller.reset()
        self.app.arrival_time = None
        self.app._height_lowered = False
        self.app.grab_time = None

        self.app.controller.active = True
        self.app.gui.walk_trigger = False

    def _handle_grab(self):
        """Grab object with arm only — no walking or height change."""
        self.app.upper_controller.reset()
        self.app.arrival_time = None
        self.app.grab_time = None
        self.app.grab_only = True

        # Start arm approach immediately using current grasp target
        T_target = getattr(self.app, 'T_dw_world', None)
        use_left = getattr(self.app, '_grasp_use_left', False)
        if T_target is not None:
            T_base_inv = np.linalg.inv(self.app.T_base_world)
            target_local = T_base_inv @ T_target
            current_mat = self.app.L_mat if use_left else self.app.R_mat
            self.app.upper_controller.start_approach(current_mat, target_local, direct=True)
            self.app._grasp_side = "left" if use_left else "right"
            side = self.app._grasp_side
            print(f"[Main] Grab-only: {side} arm approaching object...")
        else:
            print("[Main] Grab-only: no grasp target available.")
            self.app.grab_only = False

        self.app.gui.grab_trigger = False

    def _handle_auto_record(self):
        """Start automated multi-episode walk+grab recording."""
        # Generate a fresh timestamped out_dir on the first click of this
        # session if the user didn't pass --out_dir at launch (reused on later
        # clicks so episode_idx stays monotonic within a sim run).
        self._ensure_output_dir()
        total = int(getattr(self.app.gui, "auto_record_n_eps", None).value or 200)
        total = max(1, total)
        self.app._auto_rec = {
            "active": True,
            "episode": 0,
            "attempts": 0,         # success + failure, for success-rate stats
            "total": total,
            "phase": "stabilize",  # stabilize → walk → walk_done → done
            "phase_time": time.time(),
            "start_time": time.time(),
            "ep_start_time": time.time(),   # per-episode wall clock
            # Per-task override via scene_cfg["episode_timeout"]; default 60 s.
            "timeout": float(self.app.scene_cfg.get("episode_timeout", 60.0)),
        }
        print(f"[AutoRecord] Starting {self.app._auto_rec['total']}-episode auto recording (walk+grab, {self.app._auto_rec['timeout']}s timeout)...")
        self.app.gui.auto_record_trigger = False

    def _handle_stop(self):
        self.app.controller.active = False
        self.app.grab_only = False
        self.app._auto_rec = {"active": False}
        self.app.upper_controller.reset()
        self.app.arrival_time = None
        self.app._height_lowered = False
        self.app.grab_time = None
        
        self.app.action_cmd = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "height": float(self.app.controller.default_height)}
        self.app.ros_node.publish_full_command(
            vx=0.0, vy=0.0, vyaw=0.0, target_height=self.app.controller.default_height,
            L_mat=self.app.L_mat, R_mat=self.app.R_mat, head_mat=self.app.H_mat,
            left_hand=self.app.left_hand_vals, right_hand=self.app.right_hand_vals, 
            control_toggle=False
        )
        self.app.gui.stop_trigger = False

    def _handle_reset_sim(self):
        # Tear down in-progress motion so the post-silence publish doesn't send
        # stale vyaw into the freshly spawned robot.
        self.app.controller.active = False
        self.app.grab_only = False
        self.app.upper_controller.reset()
        self.app.arrival_time = None
        self.app._approach_settle_time = None
        self.app.grab_time = None
        self.app.action_cmd = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0,
                               "height": float(self.app.controller.default_height)}

        # Match auto-record: two resets spaced 1s with publishing suppressed
        # throughout, so CONTROL_GOAL_TIMEOUT (1.0s) fires before each reset
        # and planner_facing_angle_ clears. A single reset leaves yaw drift.
        now = time.time()
        self.app._pending_resets = [now + 1.0, now + 2.0]
        self.app._publish_suppressed_until = now + 2.1
        if getattr(self.app, "task_manager", None) is not None:
            self.app.task_manager.reset(self.app)
        self.app._held_object_name = None

        # Yaw randomization burst — mirrors auto_record's stabilize→yaw_burst
        # both in range *and timing*: start at +3.0 s (1 s settle after the
        # second reset at +2.0 s), 0.5 s duration. Between publish-resume
        # (+2.1 s) and burst-start the deploy sees zero commands.
        yaw_lo, yaw_hi = resolve_randomization(self.app.scene_cfg)["robot"]["yaw_deg"]
        psi_target = math.radians(random.uniform(yaw_lo, yaw_hi))
        burst_duration = 0.5
        burst_start = now + 3.0
        self.app._pending_yaw_burst = {
            "start": burst_start,
            "end": burst_start + burst_duration,
            "vyaw": psi_target / burst_duration,
            "started": False,
        }
        print(f"[ResetSim] Will rotate to yaw={math.degrees(psi_target):.1f}° after reset.")

        self.app.gui.reset_sim_trigger = False

    def _handle_task_next(self):
        self.app.gui.task_next_trigger = False
        if getattr(self.app, "task_manager", None) is None:
            print("[Task] No task manager (scene has no task_plan).")
            return
        self.app.task_manager.start_next(self.app)

    def _handle_task_run(self):
        self.app.gui.task_run_trigger = False
        if getattr(self.app, "task_manager", None) is None:
            print("[Task] No task manager (scene has no task_plan).")
            return
        self.app.task_manager.run_all(self.app)

    def _handle_task_reset(self):
        self.app.gui.task_reset_trigger = False
        if getattr(self.app, "task_manager", None) is not None:
            self.app.task_manager.reset(self.app)
            self.app._held_object_name = None
            print("[Task] Plan reset to step 1.")