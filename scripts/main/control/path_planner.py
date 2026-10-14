import numpy as np
import open3d as o3d
import heapq
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_dilation
import json
import pathlib

class PathPlanner:
    def __init__(self, ply_path, resolution=0.05, min_z=0.4, max_z=1.5, robot_radius=0.4, 
                 density_threshold=15, x_bounds=(-3.5, 3.5), y_bounds=(-6.5, 7.5)):
        self.resolution = resolution
        self.min_z = min_z       
        self.max_z = max_z       
        self.robot_radius = robot_radius
        self.density_threshold = density_threshold  

        self.blocked = 0
                
        self.x_bounds = x_bounds 
        self.y_bounds = y_bounds
        
        self.ply_path = pathlib.Path(ply_path)
        
        self.T_gs_world, self.dataparser_scale = self._load_transforms()

        print(f"[PathPlanner] Loading map from {self.ply_path}...")
        self.grid_raw, self.grid, self.grid_origin, self.grid_shape = self._generate_occupancy_map(str(self.ply_path))

        print(f"[PathPlanner] Map generated. Shape: {self.grid_shape}")

    def _load_transforms(self):
        json_path = self.ply_path.parent / "dataparser_transforms.json"
        T_gs_world = np.eye(4, dtype=np.float32)
        scale = 1.0
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            scale = float(data.get("scale", 1.0))
            tf_data = np.array(data["transform"], dtype=np.float32)
            
            if tf_data.shape == (3, 4):
                T_gs_world = np.vstack([tf_data, [0, 0, 0, 1]])
            else:
                T_gs_world = tf_data
                
        return T_gs_world, scale
    
    def _add_border_walls(self, grid, thickness_cells: int):
        t = max(1, int(thickness_cells))
        grid[:t, :] = 1
        grid[-t:, :] = 1
        grid[:, :t] = 1
        grid[:, -t:] = 1
        return grid

    def _generate_occupancy_map(self, file_path):
        pcd = o3d.io.read_point_cloud(str(file_path))
        points_gs = np.asarray(pcd.points)

        # Un-scale and transform to world
        points_unscaled = points_gs / self.dataparser_scale
        R_t = self.T_gs_world[:3, :3]
        t_t = self.T_gs_world[:3, 3]
        R_inv = np.linalg.inv(R_t)
        points_world = (R_inv @ (points_unscaled - t_t).T).T

        # --- Filter by Z height AND Room Bounds ---
        mask = (points_world[:, 2] > self.min_z) & (points_world[:, 2] < self.max_z)
        
        if self.x_bounds is not None:
            mask &= (points_world[:, 0] >= self.x_bounds[0]) & (points_world[:, 0] <= self.x_bounds[1])
        if self.y_bounds is not None:
            mask &= (points_world[:, 1] >= self.y_bounds[0]) & (points_world[:, 1] <= self.y_bounds[1])
            
        filtered_points = points_world[mask]

        # --- Force the Grid Origin and Shape to the Room Bounds ---
        if self.x_bounds is not None and self.y_bounds is not None:
            min_bounds = np.array([self.x_bounds[0], self.y_bounds[0]])
            max_bounds = np.array([self.x_bounds[1], self.y_bounds[1]])
        else:
            min_bounds = np.min(filtered_points, axis=0)[:2]
            max_bounds = np.max(filtered_points, axis=0)[:2]
            
        grid_shape = np.ceil((max_bounds - min_bounds) / self.resolution).astype(int)
        grid_origin = min_bounds

        # Project points to 2D grid
        grid_indices = np.floor((filtered_points[:, :2] - grid_origin) / self.resolution).astype(int)
        grid_indices[:, 0] = np.clip(grid_indices[:, 0], 0, grid_shape[0] - 1)
        grid_indices[:, 1] = np.clip(grid_indices[:, 1], 0, grid_shape[1] - 1)
        
        # Density Mapping
        density_grid = np.zeros(grid_shape, dtype=np.int32)
        np.add.at(density_grid, (grid_indices[:, 0], grid_indices[:, 1]), 1)
        
        grid_raw = (density_grid >= self.density_threshold).astype(np.uint8)

        inflation_cells = int(np.ceil(self.robot_radius / self.resolution))
        grid_inflated = grid_raw.copy()
        if inflation_cells > 0:
            grid_inflated = binary_dilation(grid_inflated, iterations=inflation_cells).astype(np.uint8)

        border_cells = inflation_cells + 2
        grid_inflated = self._add_border_walls(grid_inflated, border_cells)
        grid_raw = self._add_border_walls(grid_raw, 1)

        return grid_raw, grid_inflated, grid_origin, grid_shape

    def _world_to_grid(self, pos):
        idx = np.floor((pos[:2] - self.grid_origin) / self.resolution).astype(int)
        return tuple(np.clip(idx, [0, 0], np.array(self.grid_shape) - 1))

    def _grid_to_world(self, idx):
        return np.array(idx) * self.resolution + self.grid_origin + (self.resolution / 2.0)

    def _check_line_of_sight(self, start_idx, goal_idx):
        """Uses Bresenham's Line Algorithm to check if the direct straight path is obstacle-free."""
        x0, y0 = start_idx
        x1, y1 = goal_idx
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Check if out of bounds
            if not (0 <= x0 < self.grid_shape[0] and 0 <= y0 < self.grid_shape[1]):
                return False 
            # Check for an obstacle
            if self.grid[x0][y0] == 1:
                return False 
            
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
        return True
    
    def plan_path(self, start_pos, goal_pos, path_height=0.78):
        start_idx = self._world_to_grid(start_pos)
        goal_idx = self._world_to_grid(goal_pos)

        # Try a straight line first.
        if self._check_line_of_sight(start_idx, goal_idx):
            print("[PathPlanner] Direct line clear! Using straight path.")
            
            # Interpolate points along the straight line for the controller to follow
            dist = np.hypot(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])
            num_points = max(10, int(dist / self.resolution))
            
            x_vals = np.linspace(start_pos[0], goal_pos[0], num_points)
            y_vals = np.linspace(start_pos[1], goal_pos[1], num_points)
            z_vals = np.full_like(x_vals, path_height)
            
            path = np.vstack((x_vals, y_vals, z_vals)).T
            self.blocked = 0

            return path

        self.blocked = 1

        # Otherwise A*, with a cross-product tie-breaker that keeps it near the straight line.
        def heuristic(a, b):
            # Standard Euclidean distance
            h = np.hypot(a[0] - b[0], a[1] - b[1])
            
            # Tie-breaker: mathematically penalizes cells that wander away from the direct line
            dx1 = a[0] - goal_idx[0]
            dy1 = a[1] - goal_idx[1]
            dx2 = start_idx[0] - goal_idx[0]
            dy2 = start_idx[1] - goal_idx[1]
            cross = abs(dx1 * dy2 - dx2 * dy1)
            
            return h + cross * 0.001

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {start_idx: 0}
        fscore = {start_idx: heuristic(start_idx, goal_idx)}
        oheap = []
        heapq.heappush(oheap, (fscore[start_idx], start_idx))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal_idx:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_idx)
                path = path[::-1]
                
                world_path = [self._grid_to_world(idx) for idx in path]
                
                return self._smooth_path(world_path, z_height=path_height)

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                
                if 0 <= neighbor[0] < self.grid_shape[0]:
                    if 0 <= neighbor[1] < self.grid_shape[1]:
                        if self.grid[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        continue
                else:
                    continue

                move_cost = 1.414 if i != 0 and j != 0 else 1.0
                tentative_g_score = gscore[current] + move_cost

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal_idx)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        
        print("[PathPlanner] No valid path found!")
        return None

    def _smooth_path(self, world_path, z_height=0.78):
        """
        Smooths the path without aggressively cutting corners.
        Sets the final Z coordinate to the specified z_height.
        """
        if len(world_path) < 3:
            return np.pad(world_path, ((0, 0), (0, 1)), constant_values=z_height)
            
        pts = np.array(world_path).T
        
        # Spline through all the points.
        # A small 's' value allows it to smooth the zig-zags but anchors it safely 
        # to the A* grid path so it doesn't drift into walls.
        smoothing_factor = len(world_path) * 0.005 
        
        tck, u = splprep(pts, s=smoothing_factor, k=3)
        u_fine = np.linspace(0, 1, num=max(50, len(world_path)))
        smooth_pts = splev(u_fine, tck)
        
        x_coords = smooth_pts[0]
        y_coords = smooth_pts[1]
        z_coords = np.full_like(x_coords, z_height) 
        
        return np.vstack((x_coords, y_coords, z_coords)).T