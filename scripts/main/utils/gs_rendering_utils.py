import torch
import numpy as np
import pathlib
import json
from gsplat import rasterization
from plyfile import PlyData

# Factory-calibrated D435 color stream intrinsics (640x480), pulled from
# gear_sonic's live unit. Both renderers (gsplat + pyrender) accept `K` as
# (fx, fy, cx, cy); when None they fall back to fov_y with centered cx/cy.
D435_REAL_K = (607.181640625, 606.8357543945312,
               328.4209899902344, 246.59568786621094)


class VisualizerSceneManager:
    """
    Manages the Gaussian Scene poses, transformations, and rendering natively.
    """
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.model_path = pathlib.Path(model_path)
        
        print(f"[SceneManager] Loading GS from {self.model_path} natively...")
        
        (
            self.means,
            self.scales,
            self.quats,
            self.colors,
            self.opacities,
            self.sh_degree
        ) = self._load_ply_gaussians(self.model_path)
        # Cached pristine SH coefficients so color jitter can be re-applied
        # from a clean baseline (each set_color_jitter rebuilds from this).
        self._colors_orig = self.colors.clone()

        self.dataparser_scale = 1.0
        self.T_gs_world = np.eye(4, dtype=np.float32)
        
        json_path = self.model_path.parent / "dataparser_transforms.json"
        if json_path.exists():
            print(f"[SceneManager] Found transforms: {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            self.dataparser_scale = float(data.get("scale", 1.0))
            
            tf_data = np.array(data["transform"], dtype=np.float32)
            if tf_data.shape == (3, 4):
                self.T_gs_world = np.vstack([tf_data, [0, 0, 0, 1]])
            else:
                self.T_gs_world = tf_data
        else:
            print("[SceneManager] No dataparser_transforms.json. Assuming Identity.")

        self.dataparser_transform = torch.tensor(self.T_gs_world).float().to(self.device)

    def _load_ply_gaussians(self, ply_path):
        """Extracts Gaussian Splatting properties natively from a .ply file."""
        with open(ply_path, 'rb') as f:
            plydata = PlyData.read(f)
            v = plydata['vertex']

        means = torch.from_numpy(np.stack([v['x'], v['y'], v['z']], axis=-1)).float().to(self.device)

        scales = torch.from_numpy(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1)).float().to(self.device)
        scales = torch.exp(scales)  

        quats = torch.from_numpy(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1)).float().to(self.device)

        sh_components = [np.stack([v[f'f_dc_{i}'] for i in range(3)], axis=-1)]
        
        num_rest_coeffs = sum(1 for prop in v.properties if prop.name.startswith('f_rest_'))
        num_bases = num_rest_coeffs // 3
        
        for i in range(num_bases):
            sh_3chan = np.zeros((len(v.data), 3))
            for j in range(3):
                sh_3chan[:, j] = v[f'f_rest_{i + num_bases * j}']
            sh_components.append(sh_3chan)

        colors = torch.from_numpy(np.stack(sh_components, axis=1)).float().to(self.device)

        opacities = torch.from_numpy(v['opacity']).float().to(self.device)
        opacities = torch.sigmoid(opacities)  

        sh_degree = int(np.sqrt(colors.shape[1]) - 1)

        return means, scales, quats, colors, opacities, sh_degree

    # SH DC normalization: rendered_rgb ≈ 0.5 + DC_NORM * f_dc
    DC_NORM = 0.28209479177387814

    def set_color_jitter(self, mode="off", rgb_scale=(1.0, 1.0, 1.0), rgb_shift=(0.0, 0.0, 0.0)):
        """Re-derive self.colors from the cached originals with the given perturbation.

        Modes:
          - "off"               : restore originals
          - "scale_all"         : multiply every SH band by rgb_scale (natural global tint)
          - "scale_dc"          : multiply only DC band (changes base color, keeps view-dep magnitude)
          - "shift_dc"          : add rgb_shift / DC_NORM to DC band (uniform brightness shift)
          - "scale_and_shift_dc": shift_dc applied after scale_dc on the DC band
        """
        if mode == "off":
            self.colors = self._colors_orig.clone()
            return

        colors = self._colors_orig.clone()
        scale = torch.tensor(rgb_scale, device=self.device, dtype=colors.dtype).view(1, 1, 3)
        shift = torch.tensor(rgb_shift, device=self.device, dtype=colors.dtype).view(1, 1, 3) / self.DC_NORM

        if mode == "scale_all":
            colors *= scale
        elif mode == "scale_dc":
            colors[:, 0:1, :] *= scale
        elif mode == "shift_dc":
            colors[:, 0:1, :] += shift
        elif mode == "scale_and_shift_dc":
            colors[:, 0:1, :] = colors[:, 0:1, :] * scale + shift
        else:
            print(f"[SceneManager] Unknown color jitter mode: {mode!r}, leaving colors untouched.")
            return

        self.colors = colors

    def render_camera(self, T_cam_world, width, height, fov_y, K=None,
                      velocity_lin_world=None, velocity_ang_world=None):

        # 1. Calculate Intrinsics
        if K is None:
            fy = height / (2.0 * np.tan(fov_y / 2.0))
            fx = fy
            cx = (width  - 1) / 2.0
            cy = (height - 1) / 2.0
        else:
            fx, fy, cx, cy = K
        
        # 2. Format Camera Poses
        c2w = torch.from_numpy(T_cam_world).float().unsqueeze(0).to(self.device).contiguous()
        
        # Apply the dataset transform and scale
        c2w = torch.bmm(self.dataparser_transform.unsqueeze(0), c2w)
        c2w[:, :3, 3] *= self.dataparser_scale
        w2c = torch.inverse(c2w)

        # 3. Create Camera intrinsics matrix K
        K = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy

        # 4. Background Canvas (Clears the dirty buffer across loops)
        bg_color = torch.zeros(1, 4, device=self.device) 

        # 5. Render directly using gsplat
        with torch.no_grad():
            colors, _, _ = rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales,
                opacities=self.opacities,
                colors=self.colors,
                viewmats=w2c,
                Ks=K,
                width=width,
                height=height,
                radius_clip=3.0,
                sh_degree=self.sh_degree,
                render_mode='RGB+D',
                backgrounds=bg_color 
            )

        # 6. Process and Return Outputs
        depths = colors[0, ..., 3:]
        rgb = colors[0, ..., :3]
        rgb.clamp_(0, 1)

        # Convert to numpy arrays expected by your main loop
        img_out = (rgb * 255).to(torch.uint8).cpu().numpy()
        depth_out = (depths / self.dataparser_scale).cpu().numpy()

        return img_out, depth_out
    
    @staticmethod
    def colorize_depth(depth_map, min_depth=0.1, max_depth=5.0):
        """
        Colorizes a depth map (meters) for visualization.
        """
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.detach().cpu().numpy()
            
        valid_mask = (depth_map > min_depth) & (depth_map < 100.0)
        depth_norm = np.zeros_like(depth_map)
        
        if valid_mask.any():
            d_valid = depth_map[valid_mask]
            d_clamped = np.clip(d_valid, min_depth, max_depth)
            d_norm = (d_clamped - min_depth) / (max_depth - min_depth)
            depth_norm[valid_mask] = 1.0 - d_norm
            
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
        
        return depth_rgb