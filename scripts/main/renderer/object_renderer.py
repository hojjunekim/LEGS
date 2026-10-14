import numpy as np
import pyrender
import trimesh

class SimpleObjectRenderer:
    def __init__(self, viewport_width=640, viewport_height=480):
        # Initialize Scene with transparent background
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
        self.renderer = pyrender.OffscreenRenderer(viewport_width, viewport_height)

        # Node references
        self.mesh_node = None
        self.camera_node = None
        self.light_node = None

        # Setup generic Directional Light (headlight, follows camera)
        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        self.light_node = self.scene.add(dl, pose=np.eye(4))
        self._light_baseline_intensity = 1.0

    def set_lighting(self, intensity_mult=1.0, rgb=(1.0, 1.0, 1.0), ambient=0.3):
        """Set directional light intensity (× baseline), RGB color, and ambient scalar."""
        self.light_node.light.intensity = self._light_baseline_intensity * intensity_mult
        self.light_node.light.color = np.array(rgb, dtype=np.float32)
        self.scene.ambient_light = np.array([ambient, ambient, ambient], dtype=np.float32)

    def set_mesh(self, trimesh_geom, smooth=True):
        """
        Loads a Trimesh geometry into the Pyrender scene.
        """
        # Clear old mesh if it exists
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)

        # Convert Trimesh to Pyrender Mesh
        pyr_mesh = pyrender.Mesh.from_trimesh(trimesh_geom, smooth=smooth)

        # Tone down metallic factor to avoid overly dark/reflective appearance
        for p in pyr_mesh.primitives:
            mat = p.material
            if hasattr(mat, 'metallicFactor') and mat.metallicFactor > 0.5:
                mat.metallicFactor = 0.1

        self.mesh_node = self.scene.add(pyr_mesh, pose=np.eye(4))

    def set_static_mesh(self, trimesh_geom, name="static"):
        """Add a static mesh (e.g., table) that is always rendered at a fixed pose."""
        if not hasattr(self, '_static_nodes'):
            self._static_nodes = {}
        if name in self._static_nodes:
            self.scene.remove_node(self._static_nodes[name])
        pyr_mesh = pyrender.Mesh.from_trimesh(trimesh_geom)
        self._static_nodes[name] = self.scene.add(pyr_mesh, pose=np.eye(4))

    def clear_mesh(self):
        """Removes the current object from the scene."""
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
            self.mesh_node = None

    def render(self, T_cam_world, T_obj_world, scale, width, height, fov_rad, K=None):
        """
        Renders the object from the perspective of T_cam_world.
        
        Args:
            T_cam_world (4x4): Camera pose in World frame (OpenCV convention: +Z fwd, +Y down)
            T_obj_world (4x4): Object pose in World frame
            scale (float): Uniform scaling factor for the object
            width, height (int): Viewport dimensions
            fov_rad (float): Vertical FOV in radians
            
        Returns:
            color (H, W, 3): RGB image
            depth (H, W): Depth image
        """
        if self.mesh_node is None:
            return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width), dtype=np.float32)

        # 1. Update Camera
        if self.camera_node is not None:
            self.scene.remove_node(self.camera_node)
        
        if K is None:
            camera = pyrender.PerspectiveCamera(yfov=fov_rad, aspectRatio=width/height)
        else:
            fx, fy, cx, cy = K
            camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

        # Coordinate Conversion: OpenCV (Robot) -> OpenGL (Pyrender)
        # OpenCV: +Z forward, +Y down
        # OpenGL: -Z forward, +Y up
        # We rotate 180 degrees around the X-axis to convert.
        R_cv_gl = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])
        T_cam_gl = T_cam_world @ R_cv_gl
        
        self.camera_node = self.scene.add(camera, pose=T_cam_gl)

        # 2. Update Object Pose & Scale
        S = np.eye(4)
        S[0,0] = S[1,1] = S[2,2] = scale
        T_final = T_obj_world @ S
        
        self.scene.set_pose(self.mesh_node, T_final)

        # 3. Update Light to match Camera (Headlight effect)
        self.scene.set_pose(self.light_node, T_cam_gl)

        # 4. Resize Viewport if needed
        if self.renderer.viewport_width != width or self.renderer.viewport_height != height:
            self.renderer.viewport_width = width
            self.renderer.viewport_height = height

        # 5. Render
        rgba, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        
        return rgba, depth