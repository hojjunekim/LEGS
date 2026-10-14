import time
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

class RobotKinematics:
    def __init__(self, urdf_path, base_frame="pelvis", camera_names=["mid360_link"], manual_cam_offset=np.zeros(3),
                 wrist_link_names=("left_wrist_yaw_link", "right_wrist_yaw_link")):
        # A single camera name may be passed as a plain string.
        self.is_single_camera = isinstance(camera_names, str)
        self.camera_names = [camera_names] if self.is_single_camera else camera_names

        self.manual_cam_offset = manual_cam_offset
        self.prev_cam_time = time.time()
        self.prev_cam_link_world = np.eye(4, dtype=np.float32)
        self.wrist_link_names = list(wrist_link_names)
        self.wrist_frame_ids = {}

        try:
            print("[Init] Loading Pinocchio model for fast FK...")
            self.pin_model = pin.buildModelFromUrdf(urdf_path)
            self.pin_data = self.pin_model.createData()

            self.base_frame_id = self.pin_model.getFrameId(base_frame)
            self.cam_frame_ids = {name: self.pin_model.getFrameId(name) for name in self.camera_names}
            # getFrameId returns model.nframes (out-of-range) when the frame isn't found.
            for name in self.wrist_link_names:
                fid = self.pin_model.getFrameId(name)
                if fid < self.pin_model.nframes:
                    self.wrist_frame_ids[name] = fid
                else:
                    print(f"[RobotKinematics] Wrist frame '{name}' not in URDF; skipping.")

            self.joint_name_to_q_idx = {}
            for name in self.pin_model.names:
                if self.pin_model.existJointName(name):
                    j_id = self.pin_model.getJointId(name)
                    if self.pin_model.joints[j_id].idx_q >= 0:
                        self.joint_name_to_q_idx[name] = self.pin_model.joints[j_id].idx_q

            self.has_pinocchio = True
        except Exception as e:
            print(f"[Error] Pinocchio failed to load. Falling back to static offset. Error: {e}")
            self.has_pinocchio = False

    def update(self, pos, quat_wxyz, joints, primary_cam_name=None):
        if primary_cam_name is None:
            primary_cam_name = self.camera_names[0]

        # 1. Base Transform
        T_base_world = np.eye(4, dtype=np.float32)
        r_scipy = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        T_base_world[:3, :3] = r_scipy.as_matrix()
        T_base_world[:3, 3] = pos
        
        # 2. Camera Transforms (FK)
        cam_poses_world = {}
        wrist_poses_base = {}
        # Link → optical-frame rotation (x-fwd → z-fwd, OpenCV convention).
        # Skipped for frames whose URDF already terminates in an optical frame
        # (name ends in "_optical_frame"), since those are already z-fwd.
        R_offset = R.from_euler('xyz', [90, 180, 90], degrees=True).as_matrix()
        T_offset = np.eye(4, dtype=np.float32)
        T_offset[:3, :3] = R_offset

        def _post(cam_name, T):
            return T if cam_name.endswith("_optical_frame") else T @ T_offset

        if self.has_pinocchio:
            q = np.zeros(self.pin_model.nq)
            for j_name, j_val in joints.items():
                if j_name in self.joint_name_to_q_idx:
                    q[self.joint_name_to_q_idx[j_name]] = j_val

            pin.framesForwardKinematics(self.pin_model, self.pin_data, q)
            T_base_root = self.pin_data.oMf[self.base_frame_id].homogeneous
            T_root_base = np.linalg.inv(T_base_root)

            for cam_name, cam_id in self.cam_frame_ids.items():
                T_cam_root = self.pin_data.oMf[cam_id].homogeneous
                T_cam_base = T_root_base @ T_cam_root
                cam_poses_world[cam_name] = _post(cam_name, T_base_world @ T_cam_base)

            for wrist_name, wrist_id in self.wrist_frame_ids.items():
                T_wrist_root = self.pin_data.oMf[wrist_id].homogeneous
                wrist_poses_base[wrist_name] = T_root_base @ T_wrist_root
        else:
            T_cam_base = np.eye(4, dtype=np.float32)
            T_cam_base[:3, 3] = self.manual_cam_offset
            for cam_name in self.camera_names:
                cam_poses_world[cam_name] = _post(cam_name, T_base_world @ T_cam_base)

        self.last_wrist_poses_base = wrist_poses_base

        # 3. Calculate Velocities (using primary camera)
        T_primary = cam_poses_world[primary_cam_name]
        dt = time.time() - self.prev_cam_time
        if dt > 0.005:
            p_curr = T_primary[:3, 3]
            lin_vel = (p_curr - self.prev_cam_link_world[:3, 3]) / dt
            R_diff = T_primary[:3, :3] @ self.prev_cam_link_world[:3, :3].T
            ang_vel = R.from_matrix(R_diff).as_rotvec() / dt
            self.prev_cam_time = time.time()
            self.prev_cam_link_world = T_primary.copy()
        else:
            lin_vel, ang_vel = np.zeros(3), np.zeros(3)

        # Return format follows how camera_names was given.
        if self.is_single_camera:
            # Returns T_cam_world (camera-to-world transform).
            return T_base_world, cam_poses_world[self.camera_names[0]], lin_vel, ang_vel
        else:
            # Returns exactly what main_recorder expects (cam_poses_world dict)
            return T_base_world, cam_poses_world, lin_vel, ang_vel