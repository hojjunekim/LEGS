"""
Adaptive Inverse Control layer for unknown nonlinear velocity-tracking plants.

Sits between the position controller (VelocityController) and the low-level
plant (e.g. the SONIC controller). Learns the plant's input-output mapping online
from odometry feedback and inverts it to compensate for dead zones,
unknown gains, and nonlinearities.

Architecture:
    Position PD → desired_vel → [AdaptiveInverse] → commanded_vel → [Plant] → actual_vel
                                       ↑                                          │
                                       └──────── online estimation ←──────────────┘
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class AdaptiveInverse:
    """Per-axis adaptive inverse model for vx, vy, vyaw."""

    def __init__(self, dt=0.05, alpha=0.15, dead_zone_init=0.0, gain_init=1.0,
                 dead_zone_lr=0.0, gain_bounds=(0.3, 3.0), dead_zone_bounds=(0.0, 0.0)):
        self.dt = dt
        self.alpha = alpha                  # EMA smoothing for velocity estimation
        self.dead_zone_lr = dead_zone_lr    # learning rate for dead zone adaptation
        self.gain_bounds = gain_bounds
        self.dead_zone_bounds = dead_zone_bounds

        # Per-axis estimates: [vx, vy, vyaw]
        self.gain = np.array([gain_init] * 3, dtype=np.float64)
        self.dead_zone = np.array([dead_zone_init] * 3, dtype=np.float64)

        # State for velocity estimation from odometry
        self.prev_pos = None        # (x, y) world
        self.prev_yaw = None        # scalar
        self.prev_time = None

        # Last commanded values (for correlating with observed response)
        self.last_cmd = np.zeros(3)
        self.actual_vel_body = np.zeros(3)  # smoothed estimate

    def estimate_velocity(self, robot_pos, robot_quat_wxyz, timestamp):
        """Compute actual body-frame velocity from consecutive odometry."""
        r = R.from_quat([robot_quat_wxyz[1], robot_quat_wxyz[2],
                         robot_quat_wxyz[3], robot_quat_wxyz[0]])
        yaw = r.as_euler('xyz')[2]
        pos_xy = robot_pos[:2]

        if self.prev_pos is None:
            self.prev_pos = pos_xy.copy()
            self.prev_yaw = yaw
            self.prev_time = timestamp
            return np.zeros(3)

        dt = timestamp - self.prev_time
        if dt < 1e-4:
            return self.actual_vel_body

        # World-frame velocity
        d_pos = pos_xy - self.prev_pos
        d_yaw = (yaw - self.prev_yaw + np.pi) % (2 * np.pi) - np.pi

        vel_world = d_pos / dt
        vyaw_actual = d_yaw / dt

        # Rotate world velocity into body frame
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        vx_body = cos_y * vel_world[0] + sin_y * vel_world[1]
        vy_body = -sin_y * vel_world[0] + cos_y * vel_world[1]

        raw = np.array([vx_body, vy_body, vyaw_actual])

        # EMA smoothing to reduce noise
        self.actual_vel_body = (1 - self.alpha) * self.actual_vel_body + self.alpha * raw

        self.prev_pos = pos_xy.copy()
        self.prev_yaw = yaw
        self.prev_time = timestamp

        return self.actual_vel_body

    def update_model(self):
        """Update gain and dead-zone estimates from last command/response pair."""
        cmd = self.last_cmd
        actual = self.actual_vel_body

        for i in range(3):
            cmd_magnitude = abs(cmd[i])
            if cmd_magnitude < 0.01:
                # Can't learn gain from near-zero commands
                # But if we commanded ~0 and got ~0, dead zone estimate is fine
                continue

            # Predicted response under current model: actual ≈ gain * (cmd - dead_zone * sign(cmd))
            effective_cmd = cmd_magnitude - self.dead_zone[i]
            if effective_cmd < 0:
                effective_cmd = 0.0
            predicted = self.gain[i] * effective_cmd
            error = abs(actual[i]) - predicted

            # Update gain: gradient step on squared error
            if effective_cmd > 0.01:
                # d(error^2)/d(gain) = 2 * error * (-effective_cmd)
                self.gain[i] += self.dead_zone_lr * error * effective_cmd
                self.gain[i] = np.clip(self.gain[i], *self.gain_bounds)

            # Update dead zone: if we commanded and got nothing, dead zone is larger
            if cmd_magnitude > 0.05 and abs(actual[i]) < 0.02:
                # Robot didn't move despite command → dead zone is at least this big
                self.dead_zone[i] += self.dead_zone_lr * 0.5
            elif cmd_magnitude > 0.05 and abs(actual[i]) > 0.02:
                # Robot moved → dead zone is at most what we commanded
                self.dead_zone[i] -= self.dead_zone_lr * 0.2
            self.dead_zone[i] = np.clip(self.dead_zone[i], *self.dead_zone_bounds)

    def compensate(self, desired_vel):
        """
        Given desired body-frame velocity from the position controller,
        compute the actual command to send to the plant.

        desired_vel: np.array([vx, vy, vyaw]) — what the PD controller wants
        returns:     np.array([vx, vy, vyaw]) — what to actually command
        """
        commanded = np.zeros(3)
        for i in range(3):
            if abs(desired_vel[i]) < 0.005:
                # Truly want to stop — don't add dead zone compensation
                commanded[i] = 0.0
            else:
                # Invert the estimated model: cmd = desired / gain + dead_zone * sign
                commanded[i] = (desired_vel[i] / self.gain[i]) + np.sign(desired_vel[i]) * self.dead_zone[i]
        return commanded

    def process(self, desired_vx, desired_vy, desired_vyaw,
                robot_pos, robot_quat_wxyz, timestamp):
        """
        Full pipeline: estimate actual velocity → update model → compensate command.

        Call this every control step. Returns compensated (vx, vy, vyaw).
        """
        # 1. Estimate what the plant actually did last step
        self.estimate_velocity(robot_pos, robot_quat_wxyz, timestamp)

        # 2. Update the plant model from last command/response
        self.update_model()

        # 3. Compensate the new desired velocity
        desired = np.array([desired_vx, desired_vy, desired_vyaw])
        compensated = self.compensate(desired)

        # 4. Store for next iteration's model update
        self.last_cmd = compensated.copy()

        return float(compensated[0]), float(compensated[1]), float(compensated[2])

    def get_estimates(self):
        """Return current model estimates for debugging/display."""
        return {
            "gain": self.gain.copy(),
            "dead_zone": self.dead_zone.copy(),
            "actual_vel": self.actual_vel_body.copy(),
        }
