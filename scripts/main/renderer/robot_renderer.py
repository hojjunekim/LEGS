import os
import numpy as np
import trimesh
import pyrender
import yourdfpy
import pinocchio as pin

class RobotMeshRenderer:
    def __init__(self, robot_model: yourdfpy.URDF, urdf_path: str):
        self.robot_model = robot_model
        self.urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        
        self.exclude_links = ["head_link"]

        # --- 1. Pinocchio model for fast FK ---
        print(f"[MeshRenderer] Loading Pinocchio model from {urdf_path}...")
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        
        # Pre-compute joint index mapping for fast q-array building
        self.joint_name_to_q_idx = {}
        for name in self.pin_model.names:
            if self.pin_model.existJointName(name):
                j_id = self.pin_model.getJointId(name)
                if self.pin_model.joints[j_id].idx_q >= 0: 
                    self.joint_name_to_q_idx[name] = self.pin_model.joints[j_id].idx_q
                    
        # Identify the base frame ID in Pinocchio
        self.base_link_name = self.robot_model.base_link
        if self.pin_model.existFrame(self.base_link_name):
            self.base_frame_id = self.pin_model.getFrameId(self.base_link_name)
        else:
            self.base_frame_id = 0 # Fallback to root

        # --- 2. Setup Scene ---
        # Ambient light handles the shadows/dark areas
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0.0, 0.0, 0.0, 0.0])
        
        # Add Directional Light (Sunlight)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        self.light_node = self.scene.add(light, pose=np.eye(4))
        self._light_baseline_intensity = 1.0
        
        # --- 3. Load Meshes into Scene ---
        self.visual_nodes = []
        self.link_to_frame_id = {}  # link_name -> pinocchio frame ID
        
        print(f"[MeshRenderer] Loading robot meshes from {self.urdf_dir}...")
        
        for link_name, link in self.robot_model.link_map.items():

            if link_name in self.exclude_links:
                continue
                
            # Cache the Pinocchio frame ID of this link
            if self.pin_model.existFrame(link_name):
                self.link_to_frame_id[link_name] = self.pin_model.getFrameId(link_name)
            else:
                continue # Skip if pinocchio doesn't know this link

            for visual in link.visuals:
                if visual.geometry.mesh is None:
                    continue
                
                # --- Robust Mesh Loading Logic ---
                filename = visual.geometry.mesh.filename
                if os.path.isabs(filename):
                    path = filename
                else:
                    path = os.path.join(self.urdf_dir, filename)
                
                # Fix Case Sensitivity (.STL vs .stl)
                if not os.path.exists(path):
                    base, ext = os.path.splitext(path)
                    if ext == ".STL": path = base + ".stl"
                    elif ext == ".stl": path = base + ".STL"

                if not os.path.exists(path):
                    print(f"File not found: {path}")
                    continue

                try:
                    # Load Trimesh
                    tm = trimesh.load(path, force='mesh')
                    
                    # Apply Color if texture is missing
                    if visual.material and visual.material.color is not None:
                        if not hasattr(tm.visual, 'material') or tm.visual.material is None:
                            color = visual.material.color
                            if hasattr(color, 'rgba'): color = color.rgba
                            tm.visual.vertex_colors = color
                    
                    mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
                    
                    # Store the local pose to apply it during every render frame
                    local_pose = visual.origin if visual.origin is not None else np.eye(4)
                    node = self.scene.add(mesh, name=link_name, pose=local_pose)
                    
                    # Keep local_pose: render() applies it on top of the link pose every frame
                    self.visual_nodes.append({
                        'node': node,
                        'link_name': link_name,
                        'local_pose': local_pose
                    })

                except Exception as e:
                    print(f"Error loading {link_name}: {e}")

        print(f"[MeshRenderer] Successfully loaded {len(self.visual_nodes)} visual nodes into the scene.")

        # 4. Offscreen Renderer
        self.r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

    def set_lighting(self, intensity_mult=1.0, rgb=(1.0, 1.0, 1.0), ambient=0.3):
        """Set directional light intensity (× baseline), RGB color, and ambient scalar."""
        self.light_node.light.intensity = self._light_baseline_intensity * intensity_mult
        self.light_node.light.color = np.array(rgb, dtype=np.float32)
        self.scene.ambient_light = np.array([ambient, ambient, ambient], dtype=np.float32)

    def render(self, joints, T_base_world, T_cam_world, width, height, fov_y, K=None):
        # --- A. Update Robot Pose ---
        # Start from the neutral configuration (np.zeros is wrong for continuous/quaternion joints)
        q = pin.neutral(self.pin_model) 
        for j_name, j_val in joints.items():
            if j_name in self.joint_name_to_q_idx:
                q[self.joint_name_to_q_idx[j_name]] = j_val
                
        # Compute FK
        pin.framesForwardKinematics(self.pin_model, self.pin_data, q)
        
        # Get base transform and invert it once
        T_base_root = self.pin_data.oMf[self.base_frame_id].homogeneous
        T_base_root_inv = np.linalg.inv(T_base_root)
        
        for item in self.visual_nodes:
            link_name = item['link_name']
            node = item['node']
            local_pose = item['local_pose']
            
            frame_id = self.link_to_frame_id.get(link_name)
            if frame_id is not None:
                # T_link_root
                T_link_root = self.pin_data.oMf[frame_id].homogeneous
                
                # T_link_base = Base_Inv * Link
                T_lb = T_base_root_inv @ T_link_root
                
                # Final world pose: T_world_base * T_base_link * T_link_visual
                T_lw = T_base_world @ T_lb @ local_pose
                self.scene.set_pose(node, pose=T_lw)

        # --- B. Setup Camera ---
        if self.r.viewport_width != width or self.r.viewport_height != height:
            self.r.viewport_width = width
            self.r.viewport_height = height

        # Convert OpenCV Camera (Right-Down-Fwd) -> OpenGL (Right-Up-Back)
        T_cv_gl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float32)
        cam_pose = T_cam_world @ T_cv_gl
        
        if K is None:
            camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=width/height, znear=0.05, zfar=100.0)
        else:
            fx, fy, cx, cy = K
            camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100.0)
        cam_node = self.scene.add(camera, pose=cam_pose)
        
        # --- C. Render ---
        rgba, depth = self.r.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        
        # --- D. Cleanup ---
        self.scene.remove_node(cam_node)
        
        return rgba, depth