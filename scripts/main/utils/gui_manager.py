import os
import json
import pathlib
from datetime import datetime
from PIL import Image
import numpy as np
import trimesh.transformations as tft
from scipy.spatial.transform import Rotation as R
import concurrent.futures

from control.upper_body_control import (
    IDEAL_WRIST_QUAT_WXYZ,
    IDEAL_L_WRIST_POS,
    IDEAL_R_WRIST_POS,
)

def transform_to_matrix(pos, quat_wxyz):
    mat = tft.quaternion_matrix(quat_wxyz)
    mat[:3, 3] = pos
    return mat

class VisualizerGUI:
    def __init__(self, server, primary_object_cfg):
        self.server = server
        self._initial_object_cfg = primary_object_cfg

        self.walk_trigger = False
        self.grab_trigger = False
        self.auto_record_trigger = False
        self.stop_trigger = False
        self.reset_sim_trigger = False
        self.save_image_trigger = False
        self.task_next_trigger = False
        self.task_run_trigger = False
        self.task_reset_trigger = False
        self.task_label_handle = None
        self.dataset_dir = f"./dataset/images/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.frame_count = 0

        self.save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        # Live pose of the primary object — populated by process_mujoco_object
        # each tick. Walk/Grab buttons (and any non-task fallback) read these.
        self.obj_pose = (1.0, 0.0, 1.0)
        self.obj_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

        self.trigger_sync = {"pressed": False}
        self.trigger_toggle = {"pressed": False}

        # Ideal wrist pose — single source of truth lives in upper_body_control.
        self._ideal_wrist_wxyz = IDEAL_WRIST_QUAT_WXYZ
        self._ideal_L_pos = tuple(IDEAL_L_WRIST_POS)
        self._ideal_R_pos = tuple(IDEAL_R_WRIST_POS)

        self.left_wrist_handle = self.server.scene.add_transform_controls(
            "/robot/teleop/left_wrist", scale=0.15,
            position=self._ideal_L_pos, wxyz=self._ideal_wrist_wxyz,
        )
        self.right_wrist_handle = self.server.scene.add_transform_controls(
            "/robot/teleop/right_wrist", scale=0.15,
            position=self._ideal_R_pos, wxyz=self._ideal_wrist_wxyz,
        )
        self.head_handle = self.server.scene.add_transform_controls(
            "/robot/teleop/head", scale=0.15, position=(0.00, 0.00, 0.00)
        )

        # Scene-offset gizmo — drag to shift/rotate the whole scene within the
        # gsplat background. Physics/controllers stay in scene frame; the
        # renderer reads this handle and applies it only when sampling gsplat.
        # Hidden by default so it doesn't interfere with normal data collection.
        self.scene_offset_handle = self.server.scene.add_transform_controls(
            "/scene/offset", scale=0.4,
            position=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=False,
        )

        @self.scene_offset_handle.on_update
        def _(_):
            pos = self.scene_offset_handle.position
            wxyz = self.scene_offset_handle.wxyz
            yaw = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_euler("xyz")[2]
            q = R.from_euler("z", yaw).as_quat()
            new_wxyz = np.array([q[3], q[0], q[1], q[2]])
            new_pos = (float(pos[0]), float(pos[1]), 0.0)
            if not (np.allclose(pos, new_pos) and np.allclose(wxyz, new_wxyz)):
                self.scene_offset_handle.position = new_pos
                self.scene_offset_handle.wxyz = new_wxyz

        self.loop_hz_display = self.server.gui.add_number("Loop Hz", initial_value=0.0, disabled=True, step=0.1)

        self._build_teleop_folders()
        self._build_interactive_folder()
        self._build_camera_folder()
        self._build_rendering_folder()

    def reset_to_ideal_upper_body(self):
        """Reset wrist handles and hand poses to the ideal upper body pose."""
        self.head_handle.position = (0.00, 0.00, 0.00)
        self.left_wrist_handle.position = self._ideal_L_pos
        self.right_wrist_handle.position = self._ideal_R_pos
        self.head_handle.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.left_wrist_handle.wxyz = self._ideal_wrist_wxyz
        self.right_wrist_handle.wxyz = self._ideal_wrist_wxyz
        self.l_hand_pose.value = "lego"
        self.r_hand_pose.value = "lego"

    def get_hand_joints(self, pose_name, is_left_hand):
        base_action = np.array([0.0, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5])
        if pose_name == "lego": return [0.0] * 7
        if pose_name == "stretch": return (base_action if is_left_hand else -base_action).tolist()
        if pose_name == "grab": return (-base_action if is_left_hand else base_action).tolist()
        return [0.0] * 7

    def get_current_hand_values(self):
        return (self.get_hand_joints(self.l_hand_pose.value, True),
                self.get_hand_joints(self.r_hand_pose.value, False))

    def get_body_transforms(self):
        L_mat = transform_to_matrix(self.left_wrist_handle.position, self.left_wrist_handle.wxyz)
        R_mat = transform_to_matrix(self.right_wrist_handle.position, self.right_wrist_handle.wxyz)
        H_mat = transform_to_matrix(self.head_handle.position, self.head_handle.wxyz)
        return L_mat, R_mat, H_mat

    def get_scene_offset_matrix(self):
        """xy+yaw-only 4x4 transform from the scene-offset gizmo. Identity by default."""
        pos = self.scene_offset_handle.position
        wxyz = self.scene_offset_handle.wxyz
        yaw = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_euler("xyz")[2]
        mat = np.eye(4)
        mat[:3, :3] = R.from_euler("z", yaw).as_matrix()
        mat[0, 3] = pos[0]
        mat[1, 3] = pos[1]
        return mat

    def _setup_vr_link(self, name, handle, initial_pos):
        updating = {"flag": False}
        with self.server.gui.add_folder(f"{name} Controls"):
            gui_px = self.server.gui.add_slider(f"{name} X", min=-1.0, max=1.0, step=0.01, initial_value=initial_pos[0])
            gui_py = self.server.gui.add_slider(f"{name} Y", min=-1.0, max=1.0, step=0.01, initial_value=initial_pos[1])
            gui_pz = self.server.gui.add_slider(f"{name} Z", min=-1.0, max=1.0, step=0.01, initial_value=initial_pos[2])
            gui_rr = self.server.gui.add_slider(f"{name} Roll", min=-3.14, max=3.14, step=0.05, initial_value=0.0)
            gui_rp = self.server.gui.add_slider(f"{name} Pitch", min=-3.14, max=3.14, step=0.05, initial_value=0.0)
            gui_ry = self.server.gui.add_slider(f"{name} Yaw", min=-3.14, max=3.14, step=0.05, initial_value=0.0)

        @handle.on_update
        def _(_):
            if updating["flag"]:
                return
            updating["flag"] = True
            pos = handle.position
            gui_px.value = pos[0]
            gui_py.value = pos[1]
            gui_pz.value = pos[2]
            rpy = R.from_quat([handle.wxyz[1], handle.wxyz[2], handle.wxyz[3], handle.wxyz[0]]).as_euler('xyz')
            gui_rr.value = rpy[0]
            gui_rp.value = rpy[1]
            gui_ry.value = rpy[2]
            updating["flag"] = False

        def _on_pos_update(_):
            if updating["flag"]:
                return
            updating["flag"] = True
            handle.position = (gui_px.value, gui_py.value, gui_pz.value)
            updating["flag"] = False
        gui_px.on_update(_on_pos_update)
        gui_py.on_update(_on_pos_update)
        gui_pz.on_update(_on_pos_update)

        def _on_rpy_update(_):
            if updating["flag"]:
                return
            updating["flag"] = True
            q = R.from_euler('xyz', [gui_rr.value, gui_rp.value, gui_ry.value]).as_quat()
            handle.wxyz = np.array([q[3], q[0], q[1], q[2]])
            updating["flag"] = False
        gui_rr.on_update(_on_rpy_update)
        gui_rp.on_update(_on_rpy_update)
        gui_ry.on_update(_on_rpy_update)

    def _build_teleop_folders(self):
        with self.server.gui.add_folder("High-Level Control"):
            self.vx_slider = self.server.gui.add_slider("Velocity X", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
            self.vy_slider = self.server.gui.add_slider("Velocity Y", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
            self.vyaw_slider = self.server.gui.add_slider("Velocity Yaw", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
            self.height_slider = self.server.gui.add_slider("Base Height", min=0.1, max=0.88, step=0.05, initial_value=0.78)

            stop_btn = self.server.gui.add_button("🛑 Stop Base")
            @stop_btn.on_click
            def _(_):
                self.vx_slider.value = 0.0
                self.vy_slider.value = 0.0
                self.vyaw_slider.value = 0.0

            ideal_btn = self.server.gui.add_button("Ideal Upper Body Pose")
            @ideal_btn.on_click
            def _(_):
                self.reset_to_ideal_upper_body()

        with self.server.gui.add_folder("Policy Control", expand_by_default=False):
            self.policy_enabled = False
            self.policy_use_rtc = False
            self.policy_status = self.server.gui.add_markdown("Policy: **disconnected**")
            policy_btn = self.server.gui.add_button("Toggle Policy", color="blue")
            @policy_btn.on_click
            def _(_):
                self.policy_enabled = not self.policy_enabled
                state = "ENABLED" if self.policy_enabled else "DISABLED"
                print(f"[Policy] Toggled: {state}")
                self.policy_status.content = f"Policy: **{state.lower()}**"
            rtc_btn = self.server.gui.add_button("Toggle RTC / Sequential")
            @rtc_btn.on_click
            def _(_):
                self.policy_use_rtc = not self.policy_use_rtc
                mode = "RTC (async ensemble)" if self.policy_use_rtc else "Sequential (sync)"
                print(f"[Policy] Mode: {mode}")
                self.policy_status.content = f"Policy: **{mode}**"
            self.rtc_exp_weight = self.server.gui.add_slider("RTC exp weight (k)", min=0.0, max=5.0, step=0.1, initial_value=0.0)
            self.show_chunk_plot = self.server.gui.add_checkbox("Show Action Chunk Plot", initial_value=False)
            self.show_cam_frustum = self.server.gui.add_checkbox("Show Camera Frustum", initial_value=False)

        with self.server.gui.add_folder("Upper Body Control", expand_by_default=False):
            sync_btn = self.server.gui.add_button("Snap Handles to Real Robot")
            @sync_btn.on_click
            def _(_): self.trigger_sync["pressed"] = True

            reset_vr_btn = self.server.gui.add_button("Reset VR Handles")
            @reset_vr_btn.on_click
            def _(_):
                self.left_wrist_handle.position = self._ideal_L_pos
                self.right_wrist_handle.position = self._ideal_R_pos
                self.head_handle.position = (0.00, 0.00, 0.00)

                self.left_wrist_handle.wxyz = self._ideal_wrist_wxyz
                self.right_wrist_handle.wxyz = self._ideal_wrist_wxyz
                self.head_handle.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

            self._setup_vr_link("Head", self.head_handle, (0.00, 0.00, 0.00))
            self._setup_vr_link("L_Wrist", self.left_wrist_handle, self._ideal_L_pos)
            self._setup_vr_link("R_Wrist", self.right_wrist_handle, self._ideal_R_pos)

            self.l_hand_pose = self.server.gui.add_dropdown("L Hand Pose", ["lego", "stretch", "grab"], initial_value="stretch")
            self.r_hand_pose = self.server.gui.add_dropdown("R Hand Pose", ["lego", "stretch", "grab"], initial_value="stretch")

            # Live grasp-offset tuning. Initial Z comes from the primary
            # object's scene config; X and Y are pinned as GUI defaults.
            # Grasp._wrist_target_world reads these every tick.
            init_off = self._initial_object_cfg.get("grasp", {}).get(
                "offset", [-0.11, -0.03, 0.02]
            )
            with self.server.gui.add_folder("Grasp Offset", expand_by_default=False):
                self.grasp_offset_x = self.server.gui.add_slider(
                    "Grasp Offset X", min=-0.20, max=0.20, step=0.005,
                    initial_value=-0.11,
                )
                self.grasp_offset_y = self.server.gui.add_slider(
                    "Grasp Offset Y", min=-0.20, max=0.20, step=0.005,
                    initial_value=-0.03,
                )
                self.grasp_offset_z = self.server.gui.add_slider(
                    "Grasp Offset Z", min=-0.20, max=0.20, step=0.005,
                    initial_value=float(init_off[2]),
                )

    def get_grasp_offset(self):
        """Live grasp-offset [x, y, z] from GUI sliders (right-arm convention)."""
        return [self.grasp_offset_x.value,
                self.grasp_offset_y.value,
                self.grasp_offset_z.value]

    def update_vel_display(self, vx, vy, vyaw, adaptive_info=None, height=None):
        text = f"**Vel Cmd:** vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f}"
        if height is not None:
            text += f"  |  **Height:** {height:.2f} m"
        if adaptive_info is not None:
            av = adaptive_info["actual_vel"]
            text += f"\n**Actual vel:** [{av[0]:.2f},{av[1]:.2f},{av[2]:.2f}]"
        self.vel_display.content = text

    def update_task_label(self, text):
        if self.task_label_handle is not None:
            self.task_label_handle.content = f"**Task:** {text}"

    def update_live_tracking(self, obj_pos_str, lw_world, rw_world, reach_score_str="N/A"):
        self.live_tracking_display.content = f"""
    **MuJoCo Object Pose:**
    {obj_pos_str}

    **R-Wrist Target Manipulability ($w$):**
    {reach_score_str}

    **Target Left Wrist (World):**
    X: {lw_world[0]:.2f}, Y: {lw_world[1]:.2f}, Z: {lw_world[2]:.2f}

    **Target Right Wrist (World):**
    X: {rw_world[0]:.2f}, Y: {rw_world[1]:.2f}, Z: {rw_world[2]:.2f}
    """

    def save_rgb_image(self, img_array):
        if not os.path.exists(self.dataset_dir): os.makedirs(self.dataset_dir)
        filename = os.path.join(self.dataset_dir, f"frame_{self.frame_count:05d}.jpg")
        self.save_executor.submit(lambda img, f: Image.fromarray(img).save(f), img_array.copy(), filename)
        self.frame_count += 1

    def build_gsplat_calibration(self, scene_manager, gs_root_handle, rebuild_fn,
                                 scene_name=None):
        """Live gsplat calibration: drag a gizmo + scale slider so the splat
        origin/units match the simulator/robot frame.

        Viser side: gizmo drives `/scene/gs_root` (children inherit), scale
        slider rebuilds the splat handle with rescaled means/covariances.
        Renderer side: `scene_manager.dataparser_transform` and
        `dataparser_scale` are mutated in lockstep so egocentric renders
        used during data collection match what's shown in the viewer.

        Calibration is shared across all task scenes that use the same
        splat — load/save always target `<ply_dir>/<ply_stem>/default.json`.
        `scene_name` is accepted for back-compat with existing callers but
        no longer changes the file chosen (one calibration drives every
        scene). To diverge per scene, point at a different splat (`--ply`)
        with its own `default.json`.
        """
        if scene_manager is None or gs_root_handle is None or rebuild_fn is None:
            return

        import torch

        T_dp_orig = scene_manager.dataparser_transform.detach().cpu().numpy().copy()
        s_dp_orig = float(scene_manager.dataparser_scale)
        device = scene_manager.device

        gizmo = self.server.scene.add_transform_controls(
            "/gs_calib_gizmo", scale=0.4,
            position=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=False,
        )
        self.gs_calib_gizmo = gizmo

        with self.server.gui.add_folder("GSplat Calibration", expand_by_default=False):
            show_box = self.server.gui.add_checkbox("Show Gizmo", initial_value=False)
            scale_slider = self.server.gui.add_slider(
                "Scale", min=0.1, max=10.0, step=0.005, initial_value=1.0
            )
            tx_slider = self.server.gui.add_slider("X (m)", min=-5.0, max=5.0, step=0.005, initial_value=0.0)
            ty_slider = self.server.gui.add_slider("Y (m)", min=-5.0, max=5.0, step=0.005, initial_value=0.0)
            tz_slider = self.server.gui.add_slider("Z (m)", min=-5.0, max=5.0, step=0.005, initial_value=0.0)
            roll_slider = self.server.gui.add_slider("Roll (deg)",  min=-180.0, max=180.0, step=0.25, initial_value=0.0)
            pitch_slider = self.server.gui.add_slider("Pitch (deg)", min=-180.0, max=180.0, step=0.25, initial_value=0.0)
            yaw_slider = self.server.gui.add_slider("Yaw (deg)",   min=-180.0, max=180.0, step=0.25, initial_value=0.0)
            reset_btn = self.server.gui.add_button("Reset Calibration")
            print_btn = self.server.gui.add_button("Print Calibration")
            save_btn = self.server.gui.add_button("Save Calibration")
        self.gs_calib_scale = scale_slider

        ply_path = pathlib.Path(scene_manager.model_path)
        calib_dir = ply_path.parent / ply_path.stem
        # Single shared calibration per splat — scene_name no longer routes
        # to a per-scene file, so every task scene sees the same transform.
        calib_path = calib_dir / "default.json"
        load_path = calib_path

        # Two-way link between sliders and gizmo. The flag breaks the cycle:
        # gizmo.on_update writes sliders → slider.on_update would write gizmo
        # again, etc.
        slider_sync = {"flag": False}

        def _T_calib():
            pos = gizmo.position
            wxyz = gizmo.wxyz
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
            T[:3, 3] = pos
            return T

        def _apply():
            T_calib = _T_calib()
            s_calib = float(scale_slider.value)

            gs_root_handle.position = T_calib[:3, 3]
            q = R.from_matrix(T_calib[:3, :3]).as_quat()
            gs_root_handle.wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)

            T_calib_inv = np.linalg.inv(T_calib).astype(np.float32)
            T_dp_new = (T_dp_orig @ T_calib_inv).astype(np.float32)
            scene_manager.dataparser_transform = torch.tensor(T_dp_new).to(device)
            scene_manager.T_gs_world = T_dp_new
            scene_manager.dataparser_scale = s_dp_orig / s_calib

        def _push_sliders_from_gizmo():
            slider_sync["flag"] = True
            try:
                pos = gizmo.position
                wxyz = gizmo.wxyz
                rpy_deg = R.from_quat(
                    [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
                ).as_euler("xyz", degrees=True)
                tx_slider.value = float(pos[0])
                ty_slider.value = float(pos[1])
                tz_slider.value = float(pos[2])
                roll_slider.value  = float(rpy_deg[0])
                pitch_slider.value = float(rpy_deg[1])
                yaw_slider.value   = float(rpy_deg[2])
            finally:
                slider_sync["flag"] = False

        def _push_gizmo_from_sliders():
            if slider_sync["flag"]:
                return
            slider_sync["flag"] = True
            try:
                gizmo.position = np.array(
                    [tx_slider.value, ty_slider.value, tz_slider.value], dtype=np.float32
                )
                q = R.from_euler(
                    "xyz",
                    [roll_slider.value, pitch_slider.value, yaw_slider.value],
                    degrees=True,
                ).as_quat()
                gizmo.wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
            finally:
                slider_sync["flag"] = False
            _apply()

        @gizmo.on_update
        def _(_):
            _push_sliders_from_gizmo()
            _apply()

        @show_box.on_update
        def _(_):
            gizmo.visible = show_box.value

        @scale_slider.on_update
        def _(_):
            rebuild_fn(scale=float(scale_slider.value))
            _apply()

        for s in (tx_slider, ty_slider, tz_slider,
                  roll_slider, pitch_slider, yaw_slider):
            s.on_update(lambda _: _push_gizmo_from_sliders())

        @reset_btn.on_click
        def _(_):
            slider_sync["flag"] = True
            try:
                tx_slider.value = 0.0
                ty_slider.value = 0.0
                tz_slider.value = 0.0
                roll_slider.value = 0.0
                pitch_slider.value = 0.0
                yaw_slider.value = 0.0
            finally:
                slider_sync["flag"] = False
            gizmo.position = (0.0, 0.0, 0.0)
            gizmo.wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            scale_slider.value = 1.0
            rebuild_fn(scale=1.0)
            _apply()

        @print_btn.on_click
        def _(_):
            pos = gizmo.position
            wxyz = gizmo.wxyz
            rpy = R.from_quat(
                [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
            ).as_euler("xyz", degrees=True)
            print("[GSplat Calibration]")
            print(f"  translation (x, y, z): "
                  f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            print(f"  rotation (rpy deg):    "
                  f"({rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f})")
            print(f"  scale:                 {float(scale_slider.value):.4f}")

        @save_btn.on_click
        def _(_):
            pos = gizmo.position
            wxyz = gizmo.wxyz
            rpy = R.from_quat(
                [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
            ).as_euler("xyz", degrees=True)
            data = {
                "translation": [float(pos[0]), float(pos[1]), float(pos[2])],
                "rotation_rpy_deg": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
                "scale": float(scale_slider.value),
            }
            try:
                calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(calib_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"[GSplat Calib] Saved to {calib_path}")
            except Exception as e:
                print(f"[GSplat Calib] Save failed: {e}")

        if load_path.exists():
            try:
                with open(load_path, "r") as f:
                    data = json.load(f)
                pos = data.get("translation", [0.0, 0.0, 0.0])
                rpy = data.get("rotation_rpy_deg", [0.0, 0.0, 0.0])
                scale = float(data.get("scale", 1.0))

                slider_sync["flag"] = True
                try:
                    tx_slider.value = float(pos[0])
                    ty_slider.value = float(pos[1])
                    tz_slider.value = float(pos[2])
                    roll_slider.value = float(rpy[0])
                    pitch_slider.value = float(rpy[1])
                    yaw_slider.value = float(rpy[2])
                finally:
                    slider_sync["flag"] = False

                q = R.from_euler("xyz", rpy, degrees=True).as_quat()
                gizmo.position = np.array(pos, dtype=np.float32)
                gizmo.wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
                scale_slider.value = scale  # fires rebuild_fn + _apply
                _apply()
                print(f"[GSplat Calib] Loaded from {load_path}")
            except Exception as e:
                print(f"[GSplat Calib] Load failed: {e}")

    def _build_rendering_folder(self):
        with self.server.gui.add_folder("Rendering", expand_by_default=False):
            self.show_gs_box = self.server.gui.add_checkbox("Show Splats", initial_value=True)
            self.show_statics_box = self.server.gui.add_checkbox("Show Static Meshes", initial_value=False)
            self.enable_blur = self.server.gui.add_checkbox("Motion Blur", initial_value=False)
            self.fov_slider = self.server.gui.add_slider("FOV", min=30.0, max=120.0, step=0.5, initial_value=42.5)
            self.resolution_slider = self.server.gui.add_slider("Res Scale", min=0.1, max=1.0, step=0.1, initial_value=1.0)
            self.show_grid_box = self.server.gui.add_checkbox("Show Occupancy Grid", initial_value=False)

            self.show_scene_offset_box = self.server.gui.add_checkbox(
                "Show Scene Offset Gizmo", initial_value=False
            )
            reset_offset_btn = self.server.gui.add_button("Reset Scene Offset")

            @self.show_scene_offset_box.on_update
            def _(_):
                self.scene_offset_handle.visible = self.show_scene_offset_box.value

            @reset_offset_btn.on_click
            def _(_):
                self.scene_offset_handle.position = (0.0, 0.0, 0.0)
                self.scene_offset_handle.wxyz = (1.0, 0.0, 0.0, 0.0)

    def _build_camera_folder(self):
        with self.server.gui.add_folder("Camera Views"):
            self.realtime_render_box = self.server.gui.add_checkbox("Real-Time Rendering", initial_value=True)
            
            self.render_robot_box = self.server.gui.add_checkbox("Render Robot", initial_value=True)
            
            save_btn = self.server.gui.add_button("🔴 Toggle Recording")
            @save_btn.on_click
            def _(_):
                self.save_image_trigger = not self.save_image_trigger
                print(f"[Data Collection] Recording {'STARTED' if self.save_image_trigger else 'STOPPED'}.")
            
            self.d435_cam_handle = self.server.gui.add_image(image=np.zeros((100, 100, 3), dtype=np.uint8), label="D435")

    def _build_interactive_folder(self):
        with self.server.gui.add_folder("Interactive Object"):
            self.adaptive_inverse_box = self.server.gui.add_checkbox("Adaptive Inverse", initial_value=True)
            self.vel_display = self.server.gui.add_markdown("**Vel Cmd:** vx=0.00, vy=0.00, vyaw=0.00  |  **Height:** -.-- m")

            self.task_label_handle = self.server.gui.add_markdown("**Task:** (no plan)")
            self.target_dist_display = self.server.gui.add_markdown("**Pelvis→target:** —")
            self.grasp_status_display = self.server.gui.add_markdown("**Grasp check:** —")
            self.last_grasp_result_display = self.server.gui.add_markdown("**Last grasp:** —")
            task_next_btn = self.server.gui.add_button("▶ Next Task", color="green")
            @task_next_btn.on_click
            def _(_): self.task_next_trigger = True

            task_run_btn = self.server.gui.add_button("▶▶ Run Remaining Tasks", color="green")
            @task_run_btn.on_click
            def _(_): self.task_run_trigger = True

            task_reset_btn = self.server.gui.add_button("Reset Task Plan")
            @task_reset_btn.on_click
            def _(_): self.task_reset_trigger = True

            walk_btn = self.server.gui.add_button("🚀 Walk to Object", color="green")
            @walk_btn.on_click
            def _(_): self.walk_trigger = True

            grab_btn = self.server.gui.add_button("✊ Grab Object", color="blue")
            @grab_btn.on_click
            def _(_): self.grab_trigger = True

            self.auto_record_n_eps = self.server.gui.add_number(
                "Episodes to record", initial_value=200, min=1, step=1
            )
            auto_record_btn = self.server.gui.add_button("🔴 Auto Record", color="red")
            @auto_record_btn.on_click
            def _(_): self.auto_record_trigger = True

            stop_btn = self.server.gui.add_button("🛑 Stop Robot", color="red")
            @stop_btn.on_click
            def _(_): self.stop_trigger = True

            reset_sim_btn = self.server.gui.add_button("Reset MuJoCo Sim")
            @reset_sim_btn.on_click
            def _(_): self.reset_sim_trigger = True