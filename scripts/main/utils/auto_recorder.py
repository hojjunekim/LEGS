"""Auto-record mixin for VisualizerApp.

State machine for automated multi-episode recording. If the active scene has
a `task_plan`, the task manager drives the run and per-subtask Verify
primitives discard episodes that fail a physics check. Otherwise falls back
to the legacy walk+grab flow.
"""

import json
import math
import os
import random
import time

from scene_config import resolve_randomization


class AutoRecordMixin:
    """Mixin providing automated multi-episode recording."""

    def _write_auto_record_info(self, elapsed, attempts):
        out_dir = getattr(self.recorder, "output_dir", None)
        if not out_dir:
            return
        durs = self._auto_rec.get("ep_durations", [])
        total = self._auto_rec.get("total", 0)
        avg = (sum(durs) / len(durs)) if durs else 0.0
        hours, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        scene = getattr(getattr(self, "args", None), "scene", None) \
            or getattr(self, "_task_name", None)
        info = {
            "scene_name": scene,
            "num_episodes_recorded": total,
            "num_attempts": attempts,
            "success_rate": (total / attempts) if attempts > 0 else 0.0,
            "total_time_seconds": round(elapsed, 1),
            "total_time_human": f"{hours:d}h {mins:02d}m {secs:02d}s",
            "avg_episode_seconds": round(avg, 2),
            "min_episode_seconds": round(min(durs), 2) if durs else 0.0,
            "max_episode_seconds": round(max(durs), 2) if durs else 0.0,
        }
        path = os.path.join(out_dir, "_info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"[AutoRecord] Wrote summary -> {path}")

    def _auto_record_finish_episode(self, success, reason=""):
        """Common cleanup for an auto-record episode (success or failure)."""
        ar = self._auto_rec
        ep, total = ar["episode"], ar["total"]

        # Stop the walk+grab state machine
        self.controller.active = False
        self.grab_only = False
        self.upper_controller.reset()
        self.arrival_time = None
        self._approach_settle_time = None
        self.grab_time = None

        # Stop recording first — commanding default_height below would make the
        # robot start raising the pelvis, and we want the final recorded frame
        # to show the robot still at the low place-height.
        self.gui.save_image_trigger = False

        # Reset arms to ideal pose
        self.gui.reset_to_ideal_upper_body()
        self.left_hand_vals = self.gui.get_hand_joints("lego", is_left_hand=True)
        self.right_hand_vals = self.gui.get_hand_joints("lego", is_left_hand=False)

        # Stop velocity and send ideal arm pose
        self.action_cmd = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "height": float(self.controller.default_height)}
        self.ros_node.publish_full_command(
            vx=0.0, vy=0.0, vyaw=0.0, target_height=self.controller.default_height,
            L_mat=self.L_mat, R_mat=self.R_mat, head_mat=self.H_mat,
            left_hand=self.left_hand_vals, right_hand=self.right_hand_vals,
            control_toggle=False
        )

        ar["attempts"] = ar.get("attempts", 0) + 1
        ep_dur = time.time() - ar.get("ep_start_time", time.time())
        if success:
            print(
                f"[AutoRecord] Ep {ep+1}/{total}: SUCCESS in {ep_dur:.1f}s, "
                f"recording saved."
            )
            ar["episode"] += 1
            ar.setdefault("ep_durations", []).append(ep_dur)
        else:
            tag = f" ({reason})" if reason else ""
            print(
                f"[AutoRecord] Ep {ep+1}/{total}: FAILED{tag} in {ep_dur:.1f}s, "
                f"discarding episode."
            )
            self.recorder.discard_current_episode()
            # Kill is_recording synchronously. save_image_trigger = False above
            # only flips the trigger; the toggle that sets is_recording and
            # calls _save_current_episode fires next tick. Without this, the
            # current tick's record_robot_state appends one ideal-pose frame,
            # and next tick's save writes it to episode_{idx-1}.json —
            # overwriting the previous successful episode.
            self.is_recording = False
        # Running success rate: saved successes / attempts so far.
        succ = ar["episode"]
        attempts = ar["attempts"]
        rate = (100.0 * succ / attempts) if attempts > 0 else 0.0
        print(f"[AutoRecord] Success rate: {succ}/{attempts} ({rate:.1f}%)")

        # Deferred resets — fire twice to settle the deploy when the previous
        # episode ended in a yawed pose (yaw_burst integrates body+target
        # together, but the deploy's planner only re-zeros after
        # CONTROL_GOAL_TIMEOUT, so a single reset can leave residual drift).
        # Publishing stays suppressed across both resets.
        now = time.time()
        self._pending_resets = [now + 1.0, now + 2.0]
        self._publish_suppressed_until = now + 2.1
        if getattr(self, "task_manager", None) is not None:
            self.task_manager.reset(self)
        # Plans that end mid-grasp (e.g. bottle_close, bottle_to_mug) leave
        # _held_object_name set; without this, next episode's first Approach
        # would falsely think it's still holding and trigger a stepback.
        self._held_object_name = None

        if ar["episode"] >= total:
            ar["active"] = False
            elapsed = time.time() - ar.get("start_time", time.time())
            mins, secs = divmod(int(elapsed), 60)
            hours, mins = divmod(mins, 60)
            attempts = ar.get("attempts", total)
            rate = (100.0 * total / attempts) if attempts > 0 else 0.0
            print(
                f"[AutoRecord] All {total} episodes recorded! "
                f"Final success rate: {total}/{attempts} ({rate:.1f}%). "
                f"Total time: {hours:d}h {mins:02d}m {secs:02d}s ({elapsed:.1f}s)"
            )
            self._write_auto_record_info(elapsed, attempts)
        else:
            ar["phase"] = "stabilize"
            # Hold stabilize long enough for the deferred reset (1.1s) plus
            # a brief post-respawn settle before the next episode starts.
            ar["phase_time"] = time.time()
            ar["ep_start_time"] = time.time()
            print(f"[AutoRecord] Starting episode {ar['episode']+1}/{total}, stabilizing...")

    def _update_auto_record(self):
        """State machine for automated multi-episode recording.

        Uses the task manager (with Verify primitives for success detection)
        if the scene defines a task_plan; otherwise runs the legacy walk+grab
        flow off `gui.walk_trigger`.
        """
        ar = self._auto_rec
        if not ar["active"]:
            return

        elapsed = time.time() - ar["phase_time"]
        ep = ar["episode"]
        total = ar["total"]
        use_task = getattr(self, "task_manager", None) is not None

        if ar["phase"] == "stabilize":
            # Cover both queued resets (+1.0s, +2.0s) plus a post-respawn settle.
            if elapsed >= 3.0:
                # Drive yaw diversity from here instead of MuJoCo spawn: spawn is
                # always yaw=0, deploy's planner_facing_angle_ is 0 after timeout,
                # so they're in sync. A vyaw burst integrates both target and body
                # together — robot ends at a random yaw with no snap-back.
                yaw_lo, yaw_hi = resolve_randomization(self.scene_cfg)["robot"]["yaw_deg"]
                psi_target = math.radians(random.uniform(yaw_lo, yaw_hi))
                burst_duration = 0.5
                ar["yaw_burst_duration"] = burst_duration
                self.action_cmd["vyaw"] = psi_target / burst_duration
                ar["phase"] = "yaw_burst"
                ar["phase_time"] = time.time()
                print(f"[AutoRecord] Ep {ep+1}/{total}: rotating to yaw={math.degrees(psi_target):.1f}°...")

        elif ar["phase"] == "yaw_burst":
            if elapsed >= ar["yaw_burst_duration"]:
                self.action_cmd["vyaw"] = 0.0
                self.gui.save_image_trigger = True
                ar["phase"] = "recording_wait"
                ar["phase_time"] = time.time()
                print(f"[AutoRecord] Ep {ep+1}/{total}: recording started, waiting 1.25s...")

        elif ar["phase"] == "recording_wait":
            if elapsed >= 1.25:
                if use_task:
                    self.gui.task_run_trigger = True
                    ar["phase"] = "running_task"
                    ar["phase_time"] = time.time()
                    print(f"[AutoRecord] Ep {ep+1}/{total}: running task plan...")
                else:
                    ar["initial_obj_z"] = self.T_obj_world[2, 3]
                    self.gui.walk_trigger = True
                    ar["phase"] = "walk"
                    ar["phase_time"] = time.time()
                    print(f"[AutoRecord] Ep {ep+1}/{total}: walking to object...")

        elif ar["phase"] == "running_task":
            if elapsed >= ar["timeout"]:
                print(f"[AutoRecord] Ep {ep+1}/{total}: TIMEOUT after {ar['timeout']:.0f}s")
                self._auto_record_finish_episode(success=False, reason="timeout")
                return

            tm = self.task_manager
            # Plan is no longer running: either all primitives succeeded, or one failed.
            if tm.mode == "idle":
                if tm.failed():
                    print(f"[AutoRecord] Ep {ep+1}/{total}: task FAILED at step {tm.failed_at+1}")
                    self._auto_record_finish_episode(success=False, reason=f"verify@{tm.failed_at+1}")
                    return
                if tm.is_done():
                    ar["phase"] = "task_done_wait"
                    ar["phase_time"] = time.time()
                    print(f"[AutoRecord] Ep {ep+1}/{total}: task plan complete, waiting 0.3s...")

        elif ar["phase"] == "task_done_wait":
            if elapsed >= 0.3:
                print(f"[AutoRecord] Ep {ep+1}/{total}: SUCCESS (task plan completed)")
                self._auto_record_finish_episode(success=True)

        elif ar["phase"] == "walk":
            # Check timeout
            if elapsed >= ar["timeout"]:
                print(f"[AutoRecord] Ep {ep+1}/{total}: TIMEOUT after {ar['timeout']:.0f}s")
                self._auto_record_finish_episode(success=False, reason="timeout")
                return

            # Walk+grab sequence is done when controller is inactive and upper body is idle
            # (the walk flow ends by setting controller.active = False and controller.reached_goal = True)
            if not self.controller.active and self.controller.reached_goal and self.upper_controller.state == "idle":
                ar["phase"] = "walk_done_wait"
                ar["phase_time"] = time.time()
                print(f"[AutoRecord] Ep {ep+1}/{total}: walk+grab done, waiting 1s...")

        elif ar["phase"] == "walk_done_wait":
            # Brief wait after completion, then mark success
            if elapsed >= 1.0:
                print(f"[AutoRecord] Ep {ep+1}/{total}: SUCCESS (walk+grab sequence completed)")
                self._auto_record_finish_episode(success=True)
