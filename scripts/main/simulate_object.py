# cv2 must load before trimesh (imported below through viser.extras and yourdfpy). trimesh's compiled pip
# dependencies load the system libstdc++, and on hosts where that is older than the env's (Ubuntu 22.04),
# cv2 then fails with "CXXABI_1.3.15 not found". Loaded first, cv2 brings in the env's libstdc++ for all.
import cv2  # noqa: F401
import os
import time
import argparse
import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy
from scipy.spatial.transform import Rotation as R
import trimesh.transformations as tft

# --- Local Modules ---
from utils.ros_bridge import start_ros_thread
from utils.gs_rendering_utils import VisualizerSceneManager, D435_REAL_K
from renderer.robot_renderer import RobotMeshRenderer
from renderer.object_renderer import SimpleObjectRenderer
from utils.gui_manager import VisualizerGUI
from utils.scene_setup import setup_gs_splats, setup_occupancy_grid
from control.lower_body_control import VelocityController
from control.upper_body_control import UpperBodyController
from control.adaptive_inverse import AdaptiveInverse
from control.path_planner import PathPlanner
from utils.robot_kinematics import RobotKinematics
from renderer.main_render_pipeline import MainRenderPipeline
from utils.data_recorder import DataRecorder
from utils.event_handler import AppEventHandler
from scene_config import (get_scene, SCENES, _OBJECTS_DIR, missing_asset_message,
                          require_core_assets)
from utils.policy_runner import PolicyRunnerMixin
from utils.psi0_client import BACKENDS
from utils.auto_recorder import AutoRecordMixin
from utils.grasp_sequencer import GraspSequencerMixin
from task import TaskManager

# --- Configuration ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_URDF = "../../robots/g1/urdf/g1_29dof_with_hand.urdf"
DEFAULT_MAP = "flightroom"
DEFAULT_PLY = "../../data/gsplat/" + DEFAULT_MAP + ".ply" 
CAMERA_LINK_NAME = ["mid360_link", "d435_link", "d435_color_optical_frame"]
DEPTH_RENDER = False

class VisualizerApp(PolicyRunnerMixin, AutoRecordMixin, GraspSequencerMixin):
    def __init__(self, args):
        self.args = args
        self.scene_cfg = get_scene(args.scene)
        # Single primary object / static — the current pipeline handles one of each.
        # Multi-entity support in the renderer/GUI is a follow-up; ROS subscription
        # below already covers every entry in the scene.
        self.primary_object = self.scene_cfg["objects"][0]
        self.primary_static = self.scene_cfg["statics"][0] if self.scene_cfg.get("statics") else None
        self.setup_infrastructure()
        self.setup_scene()

        self.event_handler = AppEventHandler(self)

        # State variables
        self.is_recording = False
        self.current_action_name = "stand"
        self.T_obj_world = np.eye(4, dtype=np.float32)
        self.T_table_world = np.eye(4, dtype=np.float32)
        # Set by update_robot_state() once the first state arrives. Any read
        # before then (e.g. the target-distance block in run()) needs to handle
        # None.
        self.T_base_world = None
        self.T_cams_world = {}
        # Scenes without a primary static (e.g. tripod_to_tile on the floor)
        # never reach the `is_primary` branch in add_static_to_scene, so
        # default these to None to keep the render-pipeline check on
        # `self.table_viser_handle` from raising AttributeError.
        self.table_viser_handle = None
        self.table_scale = 1.0
        self.path = None
        self.action_cmd = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "height": 0.78}
        # State-only perturbation: when True, the recorder skips this tick so
        # the disturbance command never enters imitation-learning labels.
        # Toggled by Perturb (full duration) and Drive's mid-cruise window.
        self._skip_record = False
        self.pause_start_time = None
        self.frustums = {"mid360": None, "d435": None}

        # Transform defaults
        self.L_mat = np.eye(4)
        self.R_mat = np.eye(4)
        self.H_mat = np.eye(4)
        self.left_hand_vals = self.gui.get_hand_joints("lego", is_left_hand=True)
        self.right_hand_vals = self.gui.get_hand_joints("lego", is_left_hand=False)

        self.arrival_time = None
        self._approach_settle_time = None
        self.grab_only = False
        self._auto_rec = {"active": False}
        # Latch for manual single-episode Toggle Recording: armed once the task
        # plan starts running so auto-stop-on-success only fires after a real run.
        self._single_rec_task_ran = False

        # Task manager (high-level task plan). Created only if the scene defines
        # `task_plan`. Tick happens inside _update_control_and_grasp override.
        self._held_object_name = None
        task_plan = self.scene_cfg.get("task_plan")
        self.task_manager = TaskManager(task_plan) if task_plan else None

        # Policy inference (connected in setup_scene, don't overwrite)
        self._init_policy_state()

    def setup_infrastructure(self):
        self.ros_node = start_ros_thread()
        # Subscribe to a pose topic per scene entity (statics + manipulables).
        for entry in list(self.scene_cfg.get("statics", [])) + list(self.scene_cfg.get("objects", [])):
            self.ros_node.subscribe_pose(entry["name"])
        self.server = viser.ViserServer(port=self.args.port)

        self._collected_root = os.path.join(_PROJECT_ROOT, "data", "collected")
        # Bare task name (e.g. "g1_sonic_turn") is stashed and combined with
        # a timestamp only when 🔴 Auto Record is clicked, so the dir name
        # matches when recording actually starts. Full paths (absolute or
        # containing a separator) pass straight through to the recorder.
        out_dir = self.args.out_dir
        self._task_name = None
        if out_dir and not os.path.isabs(out_dir) and os.sep not in out_dir:
            self._task_name = out_dir
            out_dir = None
        self.recorder = DataRecorder(output_dir=out_dir)

    def setup_scene(self):
        self.robot_model = yourdfpy.URDF.load(self.args.urdf)
        self.viser_urdf = ViserUrdf(self.server, urdf_or_path=self.robot_model, root_node_name="/robot/urdf")

        try:
            self.scene_manager = VisualizerSceneManager(self.args.ply)
            has_gs = True
        except Exception as e:
            print(f"[Viser] WARNING: could not load the splat {self.args.ply} ({e!r}); running without it.")
            self.scene_manager = None
            has_gs = False

        robot_renderer = RobotMeshRenderer(self.robot_model, self.args.urdf)
        self.planner = PathPlanner(self.args.ply)
        self.controller = VelocityController(max_lin_vel=0.22, max_ang_vel=0.2)
        self.adaptive_inverse = AdaptiveInverse(dt=1.0 / 30)
        self.upper_controller = UpperBodyController(dt=1.0 / 30)

        self.gui = VisualizerGUI(self.server, self.primary_object)
        gs_root_handle, gs_rebuild_fn = setup_gs_splats(
            self.server, self.scene_manager, has_gs, self.gui.show_gs_box
        )
        if has_gs:
            self.gui.build_gsplat_calibration(
                self.scene_manager, gs_root_handle, gs_rebuild_fn,
                scene_name=self.args.scene,
            )
        setup_occupancy_grid(self.server, self.planner, self.gui.show_grid_box)

        # Track wrist + palm + thumb-tip FK so Viser can plot per-link axes
        # for debugging grasp alignment. RobotKinematics silently skips links
        # not present in the URDF (e.g. a no-hand variant).
        self.kinematics = RobotKinematics(
            self.args.urdf, self.args.base_frame, CAMERA_LINK_NAME,
            wrist_link_names=(
                "left_wrist_yaw_link", "right_wrist_yaw_link",
                "left_hand_palm_link", "right_hand_palm_link",
                "left_hand_thumb_2_link", "right_hand_thumb_2_link",
            ),
        )
        self.render_pipeline = MainRenderPipeline(self.scene_manager, robot_renderer, has_gs, K=D435_REAL_K)

        # Add every static prop in the scene to Viser. Each gets its own handle
        # keyed by name; poses are driven by ROS each tick. The first static
        # stays "primary" for the render pipeline's single-table handoff.
        self.static_viser_handles = {}
        self.static_scales = {}
        for idx, stat in enumerate(self.scene_cfg.get("statics", [])):
            self._add_static_to_viser(stat, is_primary=(idx == 0))

        # All scene objects (primary + extras) are treated like extra statics
        # for Viser/render purposes — their live poses still come from ROS
        # (MuJoCo physics drives them), but grasp logic stays keyed off
        # primary_object.name in process_mujoco_object, independent of this.
        # is_static=False so the Show-Static-Mesh checkbox doesn't hide them.
        for obj in self.scene_cfg.get("objects", []):
            self._add_static_to_viser(obj, is_primary=False, is_static=False)

        self.recorder.save_map_context(self.planner)

        # Connect to policy server if specified
        self._connect_policy()

    def _load_static_mesh(self, entry):
        """Load a static / non-primary object mesh from its entry's mesh root.

        Tries a .glb sibling first (textured), falls back to the configured file.
        """
        import trimesh
        from scipy.spatial.transform import Rotation as R

        root = _OBJECTS_DIR
        configured_path = os.path.join(root, entry["mesh"])
        base, _ = os.path.splitext(configured_path)
        glb_path = base + ".glb"
        load_path = glb_path if os.path.exists(glb_path) else configured_path

        loaded = trimesh.load(load_path)
        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

        if load_path.endswith('.glb'):
            rot_matrix = np.eye(4)
            rot_matrix[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()
            mesh.apply_transform(rot_matrix)

        return mesh

    def _add_static_to_viser(self, static_cfg, is_primary=False, is_static=True):
        """Load a static or non-primary manipulable mesh and add it to Viser.

        The primary (first) static is wired into the render pipeline as
        "the table" (back-compat). Non-primary statics are registered as
        extra renderers so they also appear in the egocentric image.

        ``is_static=False`` is used for non-primary manipulable objects (e.g.
        a second graspable item). Their visibility is decoupled from the
        Show-Static-Mesh checkbox so toggling tables off doesn't also hide
        graspable objects.
        """
        mesh = self._load_static_mesh(static_cfg)
        # mesh_offset re-centers the body's origin onto a feature of the mesh
        # (e.g. mug cylinder center vs. centroid-with-handle). The offset is in
        # raw mesh coords; trimesh.apply_translation runs before scaling, so
        # passing the unscaled value matches the MuJoCo geom-pos convention.
        mesh_offset = static_cfg.get("mesh_offset")
        if mesh_offset is not None:
            mesh = mesh.copy()
            mesh.apply_translation([-mesh_offset[0], -mesh_offset[1], -mesh_offset[2]])
        scale = static_cfg["scale"]
        name = static_cfg["name"]

        initial_visible = self.gui.show_statics_box.value if is_static else True
        # Seed pose from the scene config so the mesh is on the table from
        # frame one — process_table can only place it once a ROS pose arrives
        # on /simulation/pose/{name}, which for non-primary statics like
        # mid_table can be late or never (if the publisher hasn't tick'd yet),
        # leaving the handle parked at origin and effectively invisible.
        init_pos = np.asarray(static_cfg.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        init_rpy = np.asarray(static_cfg.get("rpy", [0.0, 0.0, 0.0]), dtype=np.float32)
        q_xyzw = R.from_euler("xyz", init_rpy).as_quat()
        init_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
        handle = self.server.scene.add_mesh_trimesh(
            f"/scene/{name}", mesh=mesh, scale=scale,
            position=init_pos, wxyz=init_wxyz,
            visible=initial_visible,
        )
        self.static_viser_handles[name] = handle
        self.static_scales[name] = scale
        print(
            f"[Viser] Added {'static' if is_static else 'manipulable'} '{name}' "
            f"(scale={scale}, pos={init_pos.tolist()})"
        )

        if is_static:
            @self.gui.show_statics_box.on_update
            def _(_):
                handle.visible = self.gui.show_statics_box.value

        # Collision-only entries are visible in Viser + simulated by MuJoCo, but
        # are not registered with the render pipeline so they don't appear in
        # egocentric frames recorded for training.
        render_in_image = static_cfg.get("render_in_image", True)

        if is_primary:
            # Back-compat aliases used by process_table and the render pipeline.
            self.table_mesh = mesh
            self.table_scale = scale
            self.table_viser_handle = handle
            if render_in_image:
                self.table_renderer = SimpleObjectRenderer()
                self.table_renderer.set_mesh(mesh, smooth=False)
                self.render_pipeline.set_table_renderer(self.table_renderer)
        elif render_in_image:
            extra_renderer = SimpleObjectRenderer()
            extra_renderer.set_mesh(mesh, smooth=False)
            self.render_pipeline.add_static_renderer(name, extra_renderer)

    @staticmethod
    def _apply_offset(T_offset, pos_or_T, wxyz=None):
        """Apply scene-offset to a viser pose. Pass either a 4x4 or (pos, wxyz)."""
        if wxyz is None:
            T_v = T_offset @ pos_or_T
        else:
            T = np.eye(4)
            T[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
            T[:3, 3] = pos_or_T
            T_v = T_offset @ T
        q = R.from_matrix(T_v[:3, :3]).as_quat()
        return T_v[:3, 3], np.array([q[3], q[0], q[1], q[2]])

    def process_table(self):
        """Update every static prop's pose in Viser from its ROS pose topic.
        Also caches T_table_world for the primary static and builds T_statics
        (dict: name -> (T_world, scale)) for every non-primary static used by
        the render pipeline.
        """
        self.T_statics = {}
        primary_name = self.primary_static["name"] if self.primary_static else None
        T_offset = getattr(self, "_T_offset", np.eye(4))
        for name, handle in self.static_viser_handles.items():
            pose_data = self.ros_node.get_pose(name)
            if not pose_data:
                continue
            pos, wxyz = pose_data
            if self._viz_tick:
                v_pos, v_wxyz = self._apply_offset(T_offset, pos, wxyz)
                handle.position = v_pos
                handle.wxyz = v_wxyz
                self.server.scene.add_frame(
                    f"/static/{name}_origin", position=v_pos, wxyz=v_wxyz,
                    show_axes=True, axes_length=0.15, axes_radius=0.005,
                )

            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
            T[:3, 3] = pos

            if name == primary_name:
                self.T_table_world[:] = T
            else:
                self.T_statics[name] = (T, self.static_scales[name])

    def _get_wrist_local(self, side):
        """Get wrist pose in robot-local frame from the GUI teleop handle."""
        handle = self.gui.left_wrist_handle if side == "left" else self.gui.right_wrist_handle
        T = np.eye(4)
        wxyz = handle.wxyz
        T[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
        T[:3, 3] = handle.position
        return T

    def get_camera_metadata(self):
        W = (int(640 * self.gui.resolution_slider.value) // 16) * 16
        H = (int(480 * self.gui.resolution_slider.value) // 16) * 16
        fov = np.deg2rad(self.gui.fov_slider.value)
        return {
            "mid360": {"W": W, "H": H, "fov": float(fov)},
            "d435": {"W": W, "H": H, "fov": float(fov)}
        }

    def sync_robot_to_ui(self, joints):
        self.robot_model.update_cfg(joints)
        L_link = self.robot_model.scene.graph.get("left_wrist_yaw_link")[0]
        R_link = self.robot_model.scene.graph.get("right_wrist_yaw_link")[0]
        Head_link = self.robot_model.scene.graph.get("head_link")[0] if "head_link" in self.robot_model.scene.graph.nodes else np.eye(4)

        self.gui.left_wrist_handle.position, self.gui.left_wrist_handle.wxyz = L_link[:3, 3], tft.quaternion_from_matrix(L_link)
        self.gui.right_wrist_handle.position, self.gui.right_wrist_handle.wxyz = R_link[:3, 3], tft.quaternion_from_matrix(R_link)
        self.gui.head_handle.position, self.gui.head_handle.wxyz = Head_link[:3, 3], tft.quaternion_from_matrix(Head_link)

        self.gui.trigger_sync["pressed"] = False

    def update_robot_state(self):
        state = self.ros_node.get_latest_state()
        if not state: return False

        self.robot_pos, self.robot_wxyz, self.joints = state

        if self.gui.trigger_sync["pressed"]:
            self.sync_robot_to_ui(self.joints)

        # When the policy is driving, don't overwrite arm/hand targets from GUI
        if not (self.policy is not None and self.gui.policy_enabled):
            self.L_mat, self.R_mat, self.H_mat = self.gui.get_body_transforms()
            self.left_hand_vals, self.right_hand_vals = self.gui.get_current_hand_values()

        if self.upper_controller.active:
            _, mat_override, hand_state = self.upper_controller.update()
            if mat_override is not None:
                use_left = getattr(self, '_grasp_side', 'right') == 'left'
                if use_left:
                    self.L_mat = mat_override
                    self.left_hand_vals = self.gui.get_hand_joints(hand_state, is_left_hand=True)
                    self.gui.left_wrist_handle.position = self.L_mat[:3, 3]
                    self.gui.left_wrist_handle.wxyz = tft.quaternion_from_matrix(self.L_mat)
                else:
                    self.R_mat = mat_override
                    self.right_hand_vals = self.gui.get_hand_joints(hand_state, is_left_hand=False)
                    self.gui.right_wrist_handle.position = self.R_mat[:3, 3]
                    self.gui.right_wrist_handle.wxyz = tft.quaternion_from_matrix(self.R_mat)

        # Control and grasp state machine
        self._update_control_and_grasp()

        # Deferred MuJoCo reset queue — each entry is a timestamp at which a
        # reset_simulation() call should fire. Scheduled this way so the
        # GR00T-WBC deploy's CONTROL_GOAL_TIMEOUT clears planner_facing_angle_
        # during the pre-reset silence window (otherwise the old facing target
        # snaps the freshly-spawned robot back to the pre-reset yaw).
        # Auto-record queues two resets because a single reset leaves a small
        # drift that a second reset after another silence window clears.
        pending_resets = getattr(self, "_pending_resets", None)
        if pending_resets:
            now_t = time.time()
            while pending_resets and now_t >= pending_resets[0]:
                self.ros_node.reset_simulation()
                pending_resets.pop(0)

        # Yaw burst after reset settles — set vyaw once at burst start, clear
        # once at end. Mirrors auto_record's yaw_burst phase so manual resets
        # don't always leave the robot facing the spawn yaw.
        burst = getattr(self, "_pending_yaw_burst", None)
        if burst:
            now_t = time.time()
            if now_t >= burst["end"]:
                self.action_cmd["vyaw"] = 0.0
                self._pending_yaw_burst = None
            elif now_t >= burst["start"] and not burst["started"]:
                self.action_cmd["vyaw"] = burst["vyaw"]
                burst["started"] = True

        # Publishing is gated so the deploy sees >1s of silence around a reset;
        # otherwise the stale vyaw target would drive the freshly-spawned robot
        # back toward the pre-reset yaw.
        if time.time() >= getattr(self, "_publish_suppressed_until", 0.0):
            self.ros_node.publish_full_command(
                vx=self.action_cmd["vx"], vy=self.action_cmd["vy"], vyaw=self.action_cmd["vyaw"], target_height=self.action_cmd["height"],
                L_mat=self.L_mat, R_mat=self.R_mat, head_mat=self.H_mat,
                left_hand=self.left_hand_vals, right_hand=self.right_hand_vals, control_toggle=False
            )

        if self.gui.trigger_toggle["pressed"]: self.gui.trigger_toggle["pressed"] = False

        # Kinematics Update
        self.T_base_world, self.T_cams_world, self.lin_vel, self.ang_vel = self.kinematics.update(self.robot_pos, self.robot_wxyz, self.joints)
        return True

    def _update_control_and_grasp(self):
        """Route control: task manager if a plan is running, else mixin fallback."""
        if self.task_manager is not None:
            self.gui.update_task_label(self.task_manager.label())
            if self.task_manager.mode != "idle":
                self.task_manager.tick(self)
                return
        super()._update_control_and_grasp()

    def process_mujoco_object(self):
        obj_pose_data = self.ros_node.get_pose(self.primary_object["name"])
        if not obj_pose_data:
            self.T_drw_world = None
            return

        obj_pos, obj_wxyz = obj_pose_data
        self.T_obj_world[:3, :3] = R.from_quat([obj_wxyz[1], obj_wxyz[2], obj_wxyz[3], obj_wxyz[0]]).as_matrix()
        self.T_obj_world[:3, 3] = obj_pos
        self.gui.obj_pose, self.gui.obj_wxyz = (obj_pos[0], obj_pos[1], obj_pos[2]), obj_wxyz

        T_offset = getattr(self, "_T_offset", np.eye(4))
        T_obj_center = self.T_obj_world

        if self._viz_tick:
            v_pos, v_wxyz = self._apply_offset(T_offset, T_obj_center)
            self.server.scene.add_frame("/object_origin", position=v_pos, wxyz=v_wxyz,
                                        show_axes=True, axes_length=0.15, axes_radius=0.005)

        # Grasp target offset lives in the scene's object entry. The GUI's
        # live slider (when present) takes precedence so the /desired_wrist
        # Viser axes track the same value the Approach primitive uses.
        grasp_cfg = self.primary_object.get("grasp", {})
        gui_offset = getattr(self.gui, "get_grasp_offset", lambda: None)()
        if gui_offset is not None:
            grasp_offset = np.asarray(gui_offset, dtype=np.float32)
        else:
            grasp_offset = np.asarray(grasp_cfg.get("offset", [-0.09, -0.04, 0.0]), dtype=np.float32)
        rot_euler_deg = np.asarray(grasp_cfg.get("rotation_euler", [0.0, 0.0, 0.0]), dtype=np.float32)
        rot_R = R.from_euler("xyz", rot_euler_deg, degrees=True).as_matrix()
        # Mirror for left arm: flip X and Z rotations (keep Y) so the hand chirality matches.
        rot_L = R.from_euler("xyz", [-rot_euler_deg[0], rot_euler_deg[1], -rot_euler_deg[2]], degrees=True).as_matrix()

        offset_R = np.eye(4, dtype=np.float32)
        offset_R[:3, :3] = rot_R
        offset_R[:3, 3] = grasp_offset
        self.T_drw_world = T_obj_center @ offset_R

        offset_L = np.eye(4, dtype=np.float32)
        offset_L[:3, :3] = rot_L
        # Left arm grasp sits 2cm lower than right — compensates for chirality-
        # induced wrist-to-fingertip vertical bias.
        offset_L[:3, 3] = [grasp_offset[0], -grasp_offset[1], grasp_offset[2] - 0.01]
        self.T_dlw_world = T_obj_center @ offset_L

        # Always use right arm for grasping
        self._grasp_use_left = False
        self.T_dw_world = self.T_drw_world

        if self._viz_tick:
            side_label = "L" if self._grasp_use_left else "R"
            v_pos, v_wxyz = self._apply_offset(T_offset, self.T_dw_world)
            self.server.scene.add_frame(f"/desired_wrist_{side_label}", position=v_pos, wxyz=v_wxyz,
                                        show_axes=True, axes_length=0.1, axes_radius=0.003)

    def render_scene(self):
        # Single browser-update gate. Drives add_frame / handle.pose updates in
        # process_table, process_mujoco_object, and the viser_urdf block below.
        policy_driving = self.policy is not None and self.gui.policy_enabled
        self._urdf_frame_counter = getattr(self, '_urdf_frame_counter', 0) + 1
        self._viz_tick = (not policy_driving) or (self._urdf_frame_counter % 4 == 0)

        # Viser browser updates (robot cfg, frames, live preview render) and the
        # scene-offset suppression below pause ONLY during Auto Record's capture
        # windows, to keep multi-episode throughput high. Manual 🔴 Toggle
        # Recording keeps the full live view so you can watch the motion you're
        # recording. The recorder saves no images and only raw world-frame poses,
        # so _T_offset is display-only and never affects saved episodes.
        auto_rec_active = self._auto_rec.get("active", False)

        # Scene offset gizmo: world-pose → splat-frame transform. Applied to
        # every viser-displayed pose so the scene matches the renderer composite.
        # Suppressed during Auto Record so the live preview matches world frame.
        self._T_offset = np.eye(4) if (self.is_recording and auto_rec_active) else self.gui.get_scene_offset_matrix()

        self.process_mujoco_object()
        self.process_table()

        if not (self.gui.save_image_trigger and auto_rec_active):
            if self._viz_tick:
                self.viser_urdf.update_cfg(self.joints)
                r_pos, r_wxyz = self._apply_offset(self._T_offset, self.robot_pos, self.robot_wxyz)
                self.server.scene.add_frame("/robot", position=r_pos, wxyz=r_wxyz, show_axes=False)

                # Hand-debug frames: palm + thumb-tip per side, axes drawn so
                # grasp alignment relative to the held object is easy to read.
                fk = getattr(self.kinematics, "last_wrist_poses_base", {})
                if self.T_base_world is not None and fk:
                    for link, path in (
                        ("left_hand_palm_link",     "/hand/L_palm"),
                        ("right_hand_palm_link",    "/hand/R_palm"),
                        ("left_hand_thumb_2_link",  "/hand/L_thumb"),
                        ("right_hand_thumb_2_link", "/hand/R_thumb"),
                    ):
                        T_base = fk.get(link)
                        if T_base is None:
                            continue
                        T_world = self.T_base_world @ T_base
                        h_pos, h_wxyz = self._apply_offset(self._T_offset, T_world)
                        self.server.scene.add_frame(
                            path, position=h_pos, wxyz=h_wxyz,
                            show_axes=True, axes_length=0.05, axes_radius=0.002,
                        )

            if self.gui.realtime_render_box.value:
                meta = self.get_camera_metadata()
                T_table = self.T_table_world if self.table_viser_handle is not None else None
                T_scene_offset = None if (self.is_recording and auto_rec_active) else self._T_offset
                img_d435, _ = self.render_pipeline.render_frame(
                    self.gui, self.joints, self.T_base_world, self.T_cams_world["d435_color_optical_frame"],
                    self.lin_vel, self.ang_vel, meta["d435"]["W"], meta["d435"]["H"], meta["d435"]["fov"],
                    T_table_world=T_table, table_scale=self.table_scale,
                    T_statics=getattr(self, "T_statics", None),
                    T_scene_offset=T_scene_offset,
                )

                self._last_img_mid = img_d435  # reuse d435 for policy input
                self._last_img_d435 = img_d435

                # Skip browser uploads when policy is driving
                if not policy_driving:
                    self.gui.d435_cam_handle.image = img_d435

                # Frustum: add/remove based on checkbox
                if self.gui.show_cam_frustum.value and not policy_driving:
                    cam_pos, cam_wxyz = self._apply_offset(self._T_offset, self.T_cams_world["d435_color_optical_frame"])
                    self.frustums["d435"] = self.server.scene.add_camera_frustum(
                        "/cam/d435", fov=meta["d435"]["fov"], aspect=meta["d435"]["W"]/meta["d435"]["H"], scale=0.15, image=img_d435,
                        position=cam_pos, wxyz=cam_wxyz,
                    )
                elif self.frustums["d435"]:
                    self.frustums["d435"].remove()
                    self.frustums["d435"] = None
            else:
                for k in ["mid360", "d435"]:
                    if self.frustums[k]: self.frustums[k].remove(); self.frustums[k] = None

    def run(self):
        while True:
            loop_start = time.time()

            self.event_handler.process_events()
            self._update_auto_record()
            state_updated = self.update_robot_state()

            if state_updated:
                self.render_scene()

                # _skip_record drops perturbation frames so the disturbance vyaw
                # never enters the imitation-learning labels (state still evolves
                # under the perturb command — only the (state, action) tuple is
                # withheld).
                if self.is_recording and not self._skip_record:
                    # 0=idle, 1=walking toward object, 2=arrived/lowering,
                    # 3=approaching (arm reaching), 4=grabbed object,
                    # 5=returning to hold pose, 6=done (holding object)
                    if self.controller.reached_goal:
                        status = 6
                    elif self.upper_controller.state == "returning":
                        status = 5
                    elif self.upper_controller.state == "grabbing":
                        status = 4
                    elif self.upper_controller.state == "approaching" and self.upper_controller.interp_progress >= 1.0:
                        status = 3
                    elif self.controller.at_object_location:
                        status = 2
                    elif self.controller.active and self.controller.initial_face_object_done:
                        status = 1
                    else:
                        status = 0

                    # Gripper state: continuous 0.0 (open/stretch) to 1.0 (closed/grab)
                    gripper_map = {"lego": 0.5, "stretch": 0.0, "grab": 1.0}
                    left_gripper = gripper_map.get(self.gui.l_hand_pose.value, 0.0)
                    right_gripper = gripper_map.get(self.gui.r_hand_pose.value, 0.0)
                    # Upper controller drives one hand — use its state for the active side
                    if self.upper_controller.active:
                        grasp_side = getattr(self, '_grasp_side', 'right')
                        grip_val = gripper_map.get(self.upper_controller.hand_state, 0.0)
                        if grasp_side == 'left':
                            left_gripper = grip_val
                        else:
                            right_gripper = grip_val
                    # While a policy drives the hands, record its gripper commands
                    if self.policy is not None and self.gui.policy_enabled:
                        left_gripper = float(getattr(self, 'left_gripper_state', left_gripper))
                        right_gripper = float(getattr(self, 'right_gripper_state', right_gripper))

                    # Collect live poses of every dynamic (freejoint) object.
                    object_poses = {}
                    for obj_cfg in self.scene_cfg.get("objects", []):
                        pose = self.ros_node.get_pose(obj_cfg["name"])
                        if pose is not None:
                            object_poses[obj_cfg["name"]] = pose

                    self.recorder.record_robot_state(
                        robot_pos=self.robot_pos, robot_quat_wxyz=self.robot_wxyz, joints_dict=self.joints,
                        T_d435_world=self.T_cams_world["d435_color_optical_frame"],
                        action_cmd=self.action_cmd, status=status,
                        left_gripper=left_gripper, right_gripper=right_gripper,
                        object_poses=object_poses,
                        L_mat=self.L_mat, R_mat=self.R_mat, H_mat=self.H_mat,
                    )

            # Policy execution
            self._run_policy_step()

            loop_duration = time.time() - loop_start
            self.gui.loop_hz_display.value = round(1.0 / max(loop_duration, 1e-6), 1)

            active = self.task_manager.current if self.task_manager else None
            wrist_fk = getattr(self.kinematics, "last_wrist_poses_base", {})
            held_name = getattr(self, "_grasp_object", None) or getattr(self, "_held_object_name", None)
            if active and isinstance(active.arg, str):
                # Show distance to whatever the current task is targeting
                # (e.g. plate during Place, even though the orange is held).
                target_name = active.arg
            elif held_name:
                target_name = held_name
            else:
                # No active task and nothing held: pick the manipulable closest
                # to the active grasping wrist (or to either wrist if no grasp
                # side set).
                target_name = self.primary_object["name"]
                all_objs = self.scene_cfg.get("objects", [])
                if len(all_objs) > 1 and self.T_base_world is not None and wrist_fk:
                    grasp_side = getattr(self, "_grasp_side", None)
                    candidate_links = (
                        [f"{grasp_side}_wrist_yaw_link"] if grasp_side in ("left", "right")
                        else ["left_wrist_yaw_link", "right_wrist_yaw_link"]
                    )
                    wrist_worlds = [
                        (self.T_base_world @ wrist_fk[name])[:3, 3]
                        for name in candidate_links if name in wrist_fk
                    ]
                    if wrist_worlds:
                        best_d = float("inf")
                        for obj in all_objs:
                            p = self.ros_node.get_pose(obj["name"])
                            if p is None:
                                continue
                            d = min(float(np.linalg.norm(p[0] - w)) for w in wrist_worlds)
                            if d < best_d:
                                best_d, target_name = d, obj["name"]
            target_pose = self.ros_node.get_pose(target_name)
            if target_pose is not None:
                dx = float(target_pose[0][0]) - float(self.robot_pos[0])
                dy = float(target_pose[0][1]) - float(self.robot_pos[1])
                self.gui.target_dist_display.content = f"**Pelvis→{target_name}:** {np.hypot(dx, dy):.2f} m"
            else:
                self.gui.target_dist_display.content = f"**Pelvis→{target_name}:** —"

            target_dt = 1.0 / 30  # 30 Hz main loop; controllers above assume dt = 1/30
            if target_dt - loop_duration > 0: time.sleep(target_dt - loop_duration)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF)
    parser.add_argument("--ply", type=str, default=DEFAULT_PLY,
                        help="Path to .ply gsplat. Bare name (no separator) "
                             "expands to ../../data/gsplat/<name>.ply.")
    parser.add_argument("--base_frame", type=str, default="pelvis")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir for episodes. Bare task name (no "
                             "path separator) expands to "
                             "<project>/data/collected/<name>_<timestamp>/. "
                             "Full path is used as-is. If omitted, deferred "
                             "until '🔴 Auto Record' is clicked, then "
                             "auto-generated as "
                             "<project>/data/collected/<timestamp>/.")
    parser.add_argument("--scene", type=str, default=None, help=f"Scene name from scene_config.SCENES. Available: {list(SCENES.keys())}")
    parser.add_argument("--policy_host", type=str, default=None, help="Policy server host (e.g. localhost); no policy if unset")
    parser.add_argument("--backend", type=str, default="psi0", choices=list(BACKENDS.keys()),
                        help="Policy server type: psi0 (HTTP), pi05 (websocket) or gr00t (HTTP)")
    parser.add_argument("--policy_port", type=int, default=None,
                        help="Policy server port (default per backend: "
                             + ", ".join(f"{k} {v['port']}" for k, v in BACKENDS.items()) + ")")
    parser.add_argument("--policy_prompt", type=str, default=None,
                        help="Language prompt (default: the scene's prompt)")

    args = parser.parse_args()
    if args.ply and "/" not in args.ply and os.sep not in args.ply:
        name = args.ply if args.ply.endswith(".ply") else args.ply + ".ply"
        args.ply = "../../data/gsplat/" + name
    require_core_assets()
    if not os.path.isfile(args.ply):
        raise SystemExit(missing_asset_message(args.ply))
    app = VisualizerApp(args)
    app.run()

if __name__ == "__main__":
    main()
