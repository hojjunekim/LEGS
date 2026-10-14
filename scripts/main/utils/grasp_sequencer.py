"""Grasp sequencer mixin for VisualizerApp.

Handles the walk+grab state machine including velocity control, navigation,
arm approach, grab, raise, and return-to-ideal sequences.
"""

import time
import numpy as np
import trimesh.transformations as tft


class GraspSequencerMixin:
    """Mixin providing control + grasp state machine for walk+grab sequences."""

    def _update_control_and_grasp(self):
        """Run velocity control and grasp state machine. Called from update_robot_state."""
        if self.controller.active:
            vx, vy, vyaw, height = self.controller.compute_command(self.robot_pos, self.robot_wxyz, self.gui.obj_pose)
            if self.gui.adaptive_inverse_box.value:
                vx, vy, vyaw = self.adaptive_inverse.process(vx, vy, vyaw, self.robot_pos, self.robot_wxyz, time.time())
                self.action_cmd = {"vx": float(vx), "vy": float(vy), "vyaw": float(vyaw), "height": float(height)}
                self.gui.update_vel_display(vx, vy, vyaw, self.adaptive_inverse.get_estimates(), height=height)
            else:
                self.adaptive_inverse.estimate_velocity(self.robot_pos, self.robot_wxyz, time.time())
                self.action_cmd = {"vx": float(vx), "vy": float(vy), "vyaw": float(vyaw), "height": float(height)}
                self.gui.update_vel_display(vx, vy, vyaw, height=height)

            # ---> Sequenced Upper Body Logic <---
            if self.controller.at_object_location:
                self._update_grasp_at_object()
            else:
                self.arrival_time = None

        elif self.grab_only:
            self._update_grab_only()

        elif getattr(self, "_auto_rec", {}).get("active", False):
            # Auto-record owns action_cmd (e.g. vyaw burst between stabilize
            # and recording_wait). Don't clobber it with GUI slider defaults.
            pass
        elif not (self.policy is not None and self.gui.policy_enabled):
            self.action_cmd = {"vx": self.gui.vx_slider.value, "vy": self.gui.vy_slider.value, "vyaw": self.gui.vyaw_slider.value, "height": self.gui.height_slider.value}

    def _update_grasp_at_object(self):
        """Sequenced upper body grasp logic when controller is at object location."""
        # Sequence 1: Lower body, then choose closer wrist and approach
        if self.upper_controller.state == "idle":
            if self.arrival_time is None:
                self.arrival_time = time.time()
                grasp_cfg = self.primary_object.get("grasp", {})
                grasp_height = grasp_cfg.get("height", 0.78)
                self.controller.target_height = grasp_height
                print(f"[Main] Reached object. Lowering to height {grasp_height:.2f}...")
            elif time.time() - self.arrival_time >= 1.0:
                use_left = getattr(self, '_grasp_use_left', False)
                T_target = getattr(self, 'T_dw_world', None)
                if T_target is not None:
                    T_base_inv = np.linalg.inv(self.T_base_world)
                    target_local = T_base_inv @ T_target
                    current_mat = self.L_mat if use_left else self.R_mat
                    self.upper_controller.start_approach(current_mat, target_local, direct=True)
                    self._grasp_side = "left" if use_left else "right"
                side = "left" if use_left else "right"
                print(f"[Main] Direct {side} arm approach...")

        # Sequence 2: Continuously update target in local frame, then wait + grab
        elif self.upper_controller.state == "approaching":
            T_target = getattr(self, 'T_dw_world', None)
            if T_target is not None:
                T_base_inv = np.linalg.inv(self.T_base_world)
                self.upper_controller.target_R_mat = T_base_inv @ T_target
            if self.upper_controller.interp_progress >= 1.0:
                if not hasattr(self, '_approach_settle_time') or self._approach_settle_time is None:
                    self._approach_settle_time = time.time()
                    print("[Main] Wrist at desired pose. Settling for 0.2 second...")
                elif time.time() - self._approach_settle_time >= 0.2:
                    self.upper_controller.trigger_grab()
                    self.grab_time = time.time()
                    self._approach_settle_time = None

        # Sequence 3: Wait 0.5 second after grabbing, then raise body back to default height
        elif self.upper_controller.state == "grabbing":
            if self.grab_time is not None and time.time() - self.grab_time >= 0.5:
                self.controller.target_height = self.controller.default_height
                self._raise_time = time.time()
                self.upper_controller.state = "raising"
                print(f"[Main] Grabbed. Raising body to {self.controller.default_height}...")

        # Sequence 3.5: Wait for body to raise, then return arm to ideal
        elif self.upper_controller.state == "raising":
            if time.time() - self._raise_time >= 1.0:
                use_left = getattr(self, '_grasp_side', 'right') == 'left'
                current_mat = self.L_mat if use_left else self.R_mat
                self.upper_controller.start_return_to_ideal(current_mat, use_left=use_left)

        # Sequence 4: Wait for return to finish, then release lower body control to GUI
        elif self.upper_controller.state == "returning":
            if self.upper_controller.interp_progress >= 1.0:
                print("[Main] Sequence complete. Releasing lower and upper body control to GUI.")

                # ---> 1. Sync the GUI to match the final ideal pose and hand state
                use_left = getattr(self, '_grasp_side', 'right') == 'left'
                if use_left:
                    self.gui.left_wrist_handle.position = self.L_mat[:3, 3]
                    self.gui.left_wrist_handle.wxyz = tft.quaternion_from_matrix(self.L_mat)
                    self.gui.l_hand_pose.value = "grab"
                else:
                    self.gui.right_wrist_handle.position = self.R_mat[:3, 3]
                    self.gui.right_wrist_handle.wxyz = tft.quaternion_from_matrix(self.R_mat)
                    self.gui.r_hand_pose.value = "grab"

                # ---> 2. Reset the autonomous upper body controller
                self.upper_controller.reset()

                # ---> 3. Release the lower body
                self.controller.active = False
                self.controller.reached_goal = True
                self.arrival_time = None
                self.grab_time = None

    def _update_grab_only(self):
        """Grab-only mode: arm approach -> grab -> return to ideal (no walking/height change)."""
        if self.upper_controller.state == "approaching":
            T_target = getattr(self, 'T_dw_world', None)
            if T_target is not None:
                T_base_inv = np.linalg.inv(self.T_base_world)
                self.upper_controller.target_R_mat = T_base_inv @ T_target
            if self.upper_controller.interp_progress >= 1.0:
                if not hasattr(self, '_approach_settle_time') or self._approach_settle_time is None:
                    self._approach_settle_time = time.time()
                    print("[Main] Grab-only: wrist at target. Settling 0.2s...")
                elif time.time() - self._approach_settle_time >= 0.2:
                    self.upper_controller.trigger_grab()
                    self.grab_time = time.time()
                    self._approach_settle_time = None

        elif self.upper_controller.state == "grabbing":
            if self.grab_time is not None and time.time() - self.grab_time >= 0.5:
                use_left = getattr(self, '_grasp_side', 'right') == 'left'
                current_mat = self.L_mat if use_left else self.R_mat
                self.upper_controller.start_return_to_ideal(current_mat, use_left=use_left)
                print("[Main] Grab-only: grabbed. Returning arm to ideal...")

        elif self.upper_controller.state == "returning":
            if self.upper_controller.interp_progress >= 1.0:
                use_left = getattr(self, '_grasp_side', 'right') == 'left'
                if use_left:
                    self.gui.left_wrist_handle.position = self.L_mat[:3, 3]
                    self.gui.left_wrist_handle.wxyz = tft.quaternion_from_matrix(self.L_mat)
                    self.gui.l_hand_pose.value = "grab"
                else:
                    self.gui.right_wrist_handle.position = self.R_mat[:3, 3]
                    self.gui.right_wrist_handle.wxyz = tft.quaternion_from_matrix(self.R_mat)
                    self.gui.r_hand_pose.value = "grab"
                self.upper_controller.reset()
                self.grab_only = False
                self.grab_time = None
                print("[Main] Grab-only: sequence complete.")
