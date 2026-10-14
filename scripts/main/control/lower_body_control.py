import numpy as np
from scipy.spatial.transform import Rotation as R

class VelocityController:
    def __init__(self, kp_angular=0.8, kd_angular=0.1, dt=0.02,
                 max_lin_vel=0.3, max_ang_vel=0.3):
        self.kp_angular = kp_angular
        self.kd_angular = kd_angular
        self.dt = dt

        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        
        self.default_height = 0.78
        self.target_height = self.default_height
        
        self.end_action_name = "stand"
        self.at_object_location = False
        self.direct_heading = None         

        self.reach_dist = 0.25
        self.reach_yaw_tol = 0.05
        self.initial_yaw_tol = 0.1

        self.active = False
        self.reached_goal = False
        self.path = None
        self.initial_alignment_done = False
        self.initial_face_object_done = False
        self.cruise_yaw_correct = False

        self.prev_robot_yaw = None
        self.prev_dist_final = None

        # Rate-limited xy speed magnitude. Caps step-to-step change so the
        # output is continuous from rest, through cruise, and back to 0 at
        # the goal. Tuned per-call (not per-second) so it's loop-rate agnostic;
        # 0.03 m/s/call ≈ 0.9 m/s² at the 30 Hz main loop.
        self.current_speed_xy = 0.0
        self.accel_step_xy = 0.03

    def set_path(self, path, end_action_name="stand", direct_heading=None, reach_dist=None,
                 cruise_yaw_correct=False):
        self.path = path
        self.initial_alignment_done = False
        self.initial_face_object_done = False
        self.end_action_name = end_action_name
        self.direct_heading = direct_heading
        self.at_object_location = False
        self.reached_goal = False
        # When True, cruise applies path-tangent yaw correction; when False,
        # the robot holds the initial heading and only corrects yaw inside
        # reach_dist. Opt-in per Drive call.
        self.cruise_yaw_correct = cruise_yaw_correct
        if reach_dist is not None:
            self.reach_dist = reach_dist
        
        # Set target height based on desired end pose
        self.target_height = 0.3 if self.end_action_name == "squat" else self.default_height

        self.prev_robot_yaw = None
        self.prev_dist_final = None
        self.current_speed_xy = 0.0

    @staticmethod
    def _wrap_pi(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def compute_command(self, robot_pos, robot_quat_wxyz, target_pos, target_quat_wxyz=None):
        if not self.active:
            self.current_speed_xy = 0.0
            return 0.0, 0.0, 0.0, self.target_height

        final_target = target_pos
        dx_final = final_target[0] - robot_pos[0]
        dy_final = final_target[1] - robot_pos[1]
        dist_final = np.hypot(dx_final, dy_final)

        r_robot = R.from_quat([robot_quat_wxyz[1], robot_quat_wxyz[2], robot_quat_wxyz[3], robot_quat_wxyz[0]])
        robot_yaw = r_robot.as_euler('xyz')[2]

        if self.prev_robot_yaw is None: self.prev_robot_yaw = robot_yaw
        if self.prev_dist_final is None: self.prev_dist_final = dist_final

        yaw_rate = self._wrap_pi(robot_yaw - self.prev_robot_yaw) / self.dt if self.dt > 0 else 0.0
        dist_rate = (dist_final - self.prev_dist_final) / self.dt if self.dt > 0 else 0.0

        self.prev_robot_yaw, self.prev_dist_final = robot_yaw, dist_final

        instant_yaw = np.arctan2(dy_final, dx_final)
        yaw_err_to_object = self._wrap_pi(instant_yaw - robot_yaw)
        terminal_heading = self.direct_heading if self.direct_heading is not None else instant_yaw
        yaw_err_to_terminal = self._wrap_pi(terminal_heading - robot_yaw)

        # Phase 1: Reaching the object location
        if not self.at_object_location:
            if (dist_final < self.reach_dist) and (abs(yaw_err_to_terminal) < self.reach_yaw_tol):
                self.at_object_location = True
                print(f"[Controller] Reached and faced object. Adjusting to height: {self.target_height}")
            elif dist_final < self.reach_dist:
                v_yaw = np.clip(self.kp_angular * yaw_err_to_terminal - self.kd_angular * yaw_rate, 
                                -self.max_ang_vel, self.max_ang_vel)
                return 0.0, 0.0, v_yaw, self.target_height

        # Phase 2: Action Execution Phase (Stop Base, Set Height)
        if self.at_object_location:
            self.path = None
            # Return 0 velocity, strictly enforce the target height
            return 0.0, 0.0, 0.0, self.target_height

        # --- Vector Field Calculation ---
        if self.path is not None and len(self.path) > 1:
            distances = np.linalg.norm(self.path[:, :2] - robot_pos[:2], axis=1)
            idx = np.argmin(distances)
            tangent = self.path[idx+1][:2] - self.path[idx][:2] if idx < len(self.path) - 1 else final_target[:2] - robot_pos[:2]
            
            t_mag = np.linalg.norm(tangent)
            T = tangent / t_mag if t_mag > 1e-4 else np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
            E = self.path[idx][:2] - robot_pos[:2]
        else:
            T = final_target[:2] - robot_pos[:2]
            t_mag = np.linalg.norm(T)
            T = T / t_mag if t_mag > 1e-4 else T
            E = np.array([0.0, 0.0])

        # --- Initial Alignment Phase ---
        if not self.initial_face_object_done:
            v_yaw = np.clip(self.kp_angular * yaw_err_to_object - self.kd_angular * yaw_rate, -self.max_ang_vel, self.max_ang_vel)
            if abs(yaw_err_to_object) > self.initial_yaw_tol: return 0.0, 0.0, v_yaw, self.target_height
            else: self.initial_face_object_done = True

        desired_yaw = np.arctan2(T[1], T[0])
        yaw_error_path = self._wrap_pi(desired_yaw - robot_yaw)

        if not self.initial_alignment_done:
            v_yaw = np.clip(self.kp_angular * yaw_error_path - self.kd_angular * yaw_rate, -self.max_ang_vel, self.max_ang_vel)
            if abs(yaw_error_path) > self.initial_yaw_tol: return 0.0, 0.0, v_yaw, self.target_height
            else: self.initial_alignment_done = True

        # Cruise yaw correction is opt-in per Drive call (cruise_yaw_correct).
        # Off by default — robot walks straight along the initial heading set
        # by yaw_align and only re-aligns once within reach_dist (phase 1 above).
        if self.cruise_yaw_correct:
            v_yaw = float(np.clip(
                self.kp_angular * yaw_error_path - self.kd_angular * yaw_rate,
                -self.max_ang_vel, self.max_ang_vel,
            ))
        else:
            v_yaw = 0.0

        # --- Continuous Flow Phase (rate-limited cruise-then-stop) ---
        decel_dist = 0.1        # start slowing this far from goal
        slow_dist = 0.40        # within this distance of goal, cap cruise at
                                # slow_lin_vel for a gentler final approach.
        slow_lin_vel = 0.22

        # Effective cruise speed: drop to slow_lin_vel when close to the
        # goal so the body doesn't barrel into the standoff at full speed.
        cruise_vel = slow_lin_vel if dist_final < slow_dist else self.max_lin_vel

        # Position-based target speed: ramps 0 → cruise_vel over [reach_dist, decel_dist].
        if dist_final < self.reach_dist:
            target_speed = 0.0
        elif dist_final < decel_dist:
            alpha = (dist_final - self.reach_dist) / (decel_dist - self.reach_dist)
            target_speed = alpha * cruise_vel
        else:
            target_speed = cruise_vel

        # Rate limit toward target_speed so step-to-step output is continuous.
        if target_speed > self.current_speed_xy:
            self.current_speed_xy = min(self.current_speed_xy + self.accel_step_xy, target_speed)
        else:
            self.current_speed_xy = max(self.current_speed_xy - self.accel_step_xy, target_speed)

        # Lateral correction: vy proportional to body-frame lateral offset to
        # the goal. Capped low so vx stays the dominant motion — previously k_cross=2
        # produced oscillation by giving vy comparable magnitude to vx.
        cos_y, sin_y = np.cos(robot_yaw), np.sin(robot_yaw)
        dy_body = -sin_y * dx_final + cos_y * dy_final
        v_y = float(np.clip(1.0 * dy_body, -0.05, 0.05))

        return self.current_speed_xy, v_y, v_yaw, self.target_height