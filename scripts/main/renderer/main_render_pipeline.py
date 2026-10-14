import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class MainRenderPipeline:
    def __init__(self, scene_manager, robot_renderer, has_gs, K=None):
        self.scene_manager = scene_manager
        self.robot_renderer = robot_renderer
        self.has_gs = has_gs
        # When set, all child renderers use these intrinsics (fx, fy, cx, cy)
        # instead of deriving from fov_y with centered cx/cy. Lets us match
        # the real D435's factory-calibrated principal point.
        self.K = K
        self.table_renderer = None
        # Additional static renderers keyed by body name. Kept separate from the
        # "primary" table_renderer for back-compat with offline_renderer.
        self.extra_static_renderers = {}

    def set_table_renderer(self, table_renderer):
        """Set a separate renderer for the primary static (table)."""
        self.table_renderer = table_renderer

    def add_static_renderer(self, name, renderer):
        """Register an additional static renderer (beyond the primary table)."""
        self.extra_static_renderers[name] = renderer

    @staticmethod
    def _hsv_transform_rgb(rgb, hue_shift_deg, sat_mult, val_mult):
        """Apply HSV transform to a uint8 RGB array (no alpha). In-place safe.

        Retained as a shared helper for the offline renderer's per-episode HSV
        jitter (scripts/data/offline_renderer.py).
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # OpenCV H is 0..180 (= deg/2), so convert deg shift accordingly.
        h_shift = int(round(hue_shift_deg / 2.0))
        h = hsv[..., 0].astype(np.int32) + h_shift
        hsv[..., 0] = (h % 180).astype(np.uint8)
        s = hsv[..., 1].astype(np.float32) * sat_mult
        hsv[..., 1] = np.clip(s, 0, 255).astype(np.uint8)
        v = hsv[..., 2].astype(np.float32) * val_mult
        hsv[..., 2] = np.clip(v, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def render_frame(self, gui, joints, T_base_world, T_cam_world, lin_vel, ang_vel, W, H, fov, T_table_world=None, table_scale=1.0, T_statics=None, T_scene_offset=None):
        """Render a full frame.

        `T_statics`: optional dict `{name: (T_world, scale)}` for each entry in
        `extra_static_renderers`. Each is rendered as its own RGBA+depth layer
        and composited along with robot/object/primary-table.

        `T_scene_offset`: optional 4x4 placing the scene origin within the
        gsplat frame. Only the gsplat sampling gets offset, so the background
        shifts beneath a stationary foreground. None = identity.
        """
        # 1. GS Background
        img_gs = np.zeros((H, W, 3), dtype=np.uint8)
        depth_gs = np.ones((H, W), dtype=np.float32) * 100.0

        if self.has_gs:
            T_cam_for_gs = T_cam_world if T_scene_offset is None else T_scene_offset @ T_cam_world
            img_gs, depth_gs = self.scene_manager.render_camera(
                T_cam_for_gs, W, H, fov, K=self.K,
                velocity_lin_world=lin_vel if gui.enable_blur.value else None,
                velocity_ang_world=ang_vel if gui.enable_blur.value else None
            )
            img_gs = np.ascontiguousarray(img_gs)
            depth_gs = np.ascontiguousarray(depth_gs)

        # 2. Robot
        img_rob = np.zeros((H, W, 4), dtype=np.uint8)
        depth_rob = np.zeros((H, W), dtype=np.float32)
        
        if self.robot_renderer and gui.render_robot_box.value:
            img_rob, depth_rob = self.robot_renderer.render(
                joints, T_base_world, T_cam_world, W, H, fov, K=self.K
            )

        # 3. Table (pose driven by ROS from MuJoCo)
        img_table = np.zeros((H, W, 4), dtype=np.uint8)
        depth_table = np.zeros((H, W), dtype=np.float32)

        if self.table_renderer is not None and T_table_world is not None:
            img_table, depth_table = self.table_renderer.render(
                T_cam_world, T_table_world, table_scale, W, H, fov, K=self.K
            )

        # 4. Extra statics (small_table, dish, etc.) — one layer per registered renderer
        extra_layers = []
        if T_statics and self.extra_static_renderers:
            for name, renderer in self.extra_static_renderers.items():
                entry = T_statics.get(name)
                if entry is None:
                    continue
                T_world, scale = entry
                img_s, depth_s = renderer.render(T_cam_world, T_world, scale, W, H, fov, K=self.K)
                extra_layers.append((img_s, depth_s))

        # 5. Composite all foreground layers by depth, then overlay on GS background
        # Collect all foreground RGBA + depth layers
        fg_layers = []
        if img_rob.shape[-1] == 4 and np.any(img_rob[..., 3] > 0):
            fg_layers.append((img_rob, depth_rob))
        if img_table.shape[-1] == 4 and np.any(img_table[..., 3] > 0):
            fg_layers.append((img_table, depth_table))
        for img_s, depth_s in extra_layers:
            if img_s.shape[-1] == 4 and np.any(img_s[..., 3] > 0):
                fg_layers.append((img_s, depth_s))

        final_img = img_gs.copy()
        final_depth = depth_gs.squeeze().copy()

        if fg_layers:
            # Build per-pixel closest foreground from all layers
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

            # Composite best foreground onto GS background
            fg_vis = best_rgba[..., 3] > 0
            if fg_vis.any():
                kernel = np.ones((3, 3), np.uint8)
                interior = cv2.erode(fg_vis.astype(np.uint8), kernel) > 0
                boundary = fg_vis & ~interior

                final_img[interior] = best_rgba[interior, :3]
                final_depth[interior] = best_depth[interior]

                if boundary.any():
                    fg_a = best_rgba[boundary, 3:4].astype(np.uint16)
                    fg_rgb = best_rgba[boundary, :3].astype(np.uint16)
                    bg_rgb = final_img[boundary].astype(np.uint16)
                    final_img[boundary] = (fg_rgb + (bg_rgb * (255 - fg_a)) // 255).astype(np.uint8)
                    final_depth[boundary] = best_depth[boundary]

        return np.ascontiguousarray(final_img), final_depth