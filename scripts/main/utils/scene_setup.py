import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import viser.transforms as vtf

def setup_gs_splats(server, scene_manager, has_gs, show_gs_box):
    """Build the viser splat handles. Returns (gs_root_handle, rebuild_fn).

    `rebuild_fn(scale)` re-creates the gsplat handle with means/covariances
    scaled by `scale` (relative to the originals). Translation/rotation are
    expected to be driven via `gs_root_handle.position` / `wxyz` — viser
    propagates onto the splat children automatically.
    """
    if not has_gs:
        return None, None

    T_gs_world = scene_manager.T_gs_world
    gs_scale = scene_manager.dataparser_scale
    R_gs_world = T_gs_world[:3, :3]
    t_gs_world = T_gs_world[:3, 3]

    means = scene_manager.means.cpu().numpy()
    scales = scene_manager.scales.cpu().numpy()
    quats = scene_manager.quats.cpu().numpy()
    opacities = scene_manager.opacities.cpu().numpy().reshape(-1, 1)

    # Extract base RGB colors from the 0th Spherical Harmonic (DC component)
    # Formula: 0.5 + SH_C0 * DC_component
    SH_C0 = 0.28209479177387814
    sh_dc = scene_manager.colors[:, 0, :].cpu().numpy()
    colors = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0)

    means_scaled = means / gs_scale
    scales_scaled = scales / gs_scale
    means_world = means_scaled @ R_gs_world - t_gs_world

    R_local = vtf.SO3(quats).as_matrix()
    S_diag = np.eye(3)[None, :, :] * (scales_scaled[:, None, :] ** 2)
    cov_local = np.einsum('nij,njk,nlk->nil', R_local, S_diag, R_local)
    cov_world = np.einsum('ij,njk,lk->nil', R_gs_world, cov_local, R_gs_world)

    gs_root_handle = server.scene.add_frame(
        "/scene/gs_root", position=np.zeros(3), wxyz=np.array([1., 0., 0., 0.]),
        show_axes=False,
    )

    state = {"handle": None}

    def _rebuild(scale=1.0):
        s = float(scale)
        if state["handle"] is not None:
            state["handle"].remove()
        state["handle"] = server.scene.add_gaussian_splats(
            "/scene/gs_root/splats",
            centers=means_world * s,
            rgbs=colors,
            opacities=opacities,
            covariances=cov_world * (s * s),
            visible=show_gs_box.value,
        )

    _rebuild(scale=1.0)

    @show_gs_box.on_update
    def _(_):
        if state["handle"] is not None:
            state["handle"].visible = show_gs_box.value

    return gs_root_handle, _rebuild
def get_random_free_position(planner, robot_pos, height=1.0, min_distance=2.0, max_distance=6.0):
    """
    Uses the PathPlanner's 2D occupancy grid to find a random valid (x, y) free space
    between min_distance and max_distance of the robot, returns it at the specified 
    height along with a quaternion making the object face the robot.
    """
    def calculate_facing_quat(obj_x, obj_y):
        # Calculate angle to face the robot
        dx = robot_pos[0] - obj_x
        dy = robot_pos[1] - obj_y
        yaw = np.arctan2(dy, dx) + np.pi/2
        
        # Convert to scipy quaternion [x, y, z, w]
        q = R.from_euler('z', yaw).as_quat()
        
        # Return as Viser format [w, x, y, z]
        return np.array([q[3], q[0], q[1], q[2]])

    def fallback():
        # Uniformly sample within an annulus (ring) between min_distance and max_distance
        angle = np.random.uniform(0, 2 * np.pi)
        
        # This math ensures a uniform distribution across the area of the ring
        u = np.random.uniform(0, 1)
        r = np.sqrt(u * (max_distance**2 - min_distance**2) + min_distance**2) 
        
        x = robot_pos[0] + r * np.cos(angle)
        y = robot_pos[1] + r * np.sin(angle)
        return (x, y, height), calculate_facing_quat(x, y)

    if planner is None or planner.grid is None:
        print(f"[Warning] No planner provided. Using fallback ({min_distance}m - {max_distance}m).")
        return fallback()

    try:
        # Find all grid indices where the space is free (grid == 0)
        free_indices = np.argwhere(planner.grid == 0)
        
        if len(free_indices) == 0:
            print("[Warning] Occupancy grid has no free space! Using fallback.")
            return fallback()
            
        # Shuffle the indices to randomly sample
        np.random.shuffle(free_indices)
        
        # Rejection sampling: check up to 2000 random free spots
        for chosen_idx in free_indices[:2000]:
            world_pos = planner._grid_to_world(chosen_idx)
            x, y = world_pos[0], world_pos[1]
            
            # Check if the Euclidean distance is within the min and max bounds
            dist = np.hypot(x - robot_pos[0], y - robot_pos[1])
            if min_distance <= dist <= max_distance:
                return (x, y, height), calculate_facing_quat(x, y)
                
        # If we loop 2000 times without finding a spot, the valid area might be blocked
        print(f"[Warning] Could not find free space between {min_distance}m and {max_distance}m. Using fallback.")
        return fallback()
        
    except Exception as e:
        print(f"[Error] Failed to sample from occupancy map: {e}")
        return fallback()
    
def setup_occupancy_grid(server, planner, show_grid_box, z_height=0.05):
    """
    Creates a point cloud from the PathPlanner's occupancy grid and binds 
    its visibility to a Viser GUI checkbox.
    """
    if planner is None or planner.grid is None:
        print("[Warning] No planner provided, skipping occupancy grid visualization.")
        return

    occupied_indices = np.argwhere(planner.grid_raw == 1)
    
    if len(occupied_indices) == 0:
        print("[Warning] Occupancy grid is empty, nothing to visualize.")
        return

    points_world = []
    for idx in occupied_indices:
        # Convert grid index to real-world coordinates
        world_pos = planner._grid_to_world(idx)
        points_world.append([world_pos[0], world_pos[1], z_height])
        
    points_world = np.array(points_world)
    
    # Color all obstacles bright red for high visibility
    colors = np.zeros_like(points_world)
    colors[:, 0] = 255  
    
    # Parent under /scene/gs_root so the grid inherits the gsplat calibration
    # transform (set on gs_root by build_gsplat_calibration). Both the grid
    # points and the splat geometry are in the raw .ply frame, so the shared
    # gs_root transform registers them together.
    grid_handle = server.scene.add_point_cloud(
        "/scene/gs_root/occupancy_grid",
        points=points_world,
        colors=colors,
        point_size=planner.resolution * 0.8,
        visible=show_grid_box.value  # Set initial visibility based on the checkbox
    )
    
    # Bind the checkbox toggle to the point cloud visibility
    @show_grid_box.on_update
    def _(_): 
        grid_handle.visible = show_grid_box.value