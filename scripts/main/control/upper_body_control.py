import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# Ideal / resting wrist pose in the pelvis-relative base frame.
# Identity orientation (palm forward), arms at a natural hanging pose.
IDEAL_WRIST_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
IDEAL_L_WRIST_POS = np.array([0.15,  0.2, 0.12])
IDEAL_R_WRIST_POS = np.array([0.15, -0.2, 0.12])

# Wrist interpolation speed (fraction of progress per second; 0→1 in 1/speed s).
# Single source of truth for the object sim; tweak here.
WRIST_INTERP_SPEED = 0.7
# Speed for retracting the wrist back to the idle/resting pose. Matches the
# approach speed so reach and pull-back motions are symmetric.
WRIST_RETURN_SPEED = 0.7

# Time to hold wrist still while the hand opens before any arm motion begins.
# Avoids fingers colliding with the object during the approach.
HAND_OPEN_DURATION = 0.5


class UpperBodyController:
    def __init__(self, dt=0.05, interp_speed=WRIST_INTERP_SPEED, return_speed=WRIST_RETURN_SPEED):
        self.dt = dt
        self.interp_speed = interp_speed
        self.return_speed = return_speed
        # Speed for the active segment. `continue_to` may override this for a
        # single segment (e.g. Grasp's plunge runs faster than the hover); the
        # next `start_approach` / `start_return_to_ideal` resets it.
        self._active_interp_speed = interp_speed

        self.active = False
        self.state = "idle" # States: idle, lifting, approaching, grabbing, returning
        self.hand_state = "grab"

        self.current_L_mat = np.eye(4)
        self.current_R_mat = np.eye(4)
        self.start_R_mat = np.eye(4)
        self.target_R_mat = np.eye(4)
        self.interp_progress = 0.0

        # Intermediate waypoint for lifting phase
        self._lift_target = np.eye(4)

        # Wall-clock time of the last update() that advanced progress. Used to
        # derive a real-time dt so motion speed is loop-rate-agnostic (the main
        # loop runs much slower than 30 Hz when realtime gsplat render is on,
        # which made the arm visibly slower in manual mode than auto-record).
        self._last_update_t = None

    def start_approach(self, current_R_mat, target_R_mat_local, direct=False, hand_state="stretch"):
        self.active = True
        self.hand_state = hand_state
        self.start_R_mat = current_R_mat.copy()
        self.target_R_mat = target_R_mat_local.copy()
        self.current_R_mat = current_R_mat.copy()
        self._active_interp_speed = self.interp_speed

        # Pre-compute lift target so _begin_motion() can transition without re-derivation.
        self._post_open_direct = direct
        if not direct:
            self._lift_clearance = 0.10
            self._lift_target = current_R_mat.copy()
            self._lift_target[2, 3] = target_R_mat_local[2, 3] + self._lift_clearance
            self._lift_target[:3, :3] = target_R_mat_local[:3, :3].copy()

        # Open the hand first; arm holds its pose. _begin_motion() runs after the delay.
        self.state = "opening"
        self._opening_start_time = time.time()
        self.interp_progress = 0.0
        self._last_update_t = None
        print(f"[UpperBody] Opening hand to '{hand_state}' before arm motion...")

    def _begin_motion(self):
        self._last_update_t = None
        if self._post_open_direct:
            self.state = "approaching"
            self.interp_progress = 0.0
            print("[UpperBody] Hand open. Direct approach to target...")
        else:
            self.state = "lifting"
            self.interp_progress = 0.0
            print(f"[UpperBody] Hand open. Lifting + orienting wrist (z={self.start_R_mat[2,3]:.3f} -> {self._lift_target[2,3]:.3f}, target={self.target_R_mat[2,3]:.3f})...")

    def _start_lateral_approach(self):
        """Transition from lifting to lateral approach."""
        self.state = "approaching"
        self.start_R_mat = self.current_R_mat.copy()
        # target_R_mat already set from start_approach
        self.interp_progress = 0.0
        print("[UpperBody] Lifted. Now approaching target XY + rotation...")

    def continue_to(self, new_target_local, speed=None):
        """Chain a second segment onto the current approach without re-opening
        the hand. Used by Grasp to transition from the hover pose to the
        plunge target as a single uninterrupted motion. Optional `speed`
        overrides the interp speed for this segment only."""
        self.start_R_mat = self.current_R_mat.copy()
        self.target_R_mat = new_target_local.copy()
        self.interp_progress = 0.0
        self.state = "approaching"
        self._last_update_t = None
        if speed is not None:
            self._active_interp_speed = speed

    def trigger_grab(self):
        self.state = "grabbing"
        self.hand_state = "grab"
        print("[UpperBody] Target reached, triggering grab!")

    def start_return_to_ideal(self, current_R_mat, use_left=False):
        """Triggered 1 second after grabbing, returns arm to resting position."""
        self.state = "returning"
        self.hand_state = "grab" # Keep grabbing the object!

        self.start_R_mat = current_R_mat.copy()

        self.target_R_mat = np.eye(4)
        self.target_R_mat[:3, 3] = IDEAL_L_WRIST_POS if use_left else IDEAL_R_WRIST_POS

        self.current_R_mat = current_R_mat.copy()
        self.interp_progress = 0.0
        self._last_update_t = None
        self._active_interp_speed = self.return_speed
        print("[UpperBody] Returning to ideal pose with object...")

    def reset(self):
        self.active = False
        self.state = "idle"
        self.hand_state = "grab"
        self._last_update_t = None

    def _interpolate(self, start_mat, target_mat):
        """LERP position + SLERP rotation using self.interp_progress."""
        pos_start = start_mat[:3, 3]
        pos_target = target_mat[:3, 3]
        pos_current = pos_start + (pos_target - pos_start) * self.interp_progress

        rot_start = R.from_matrix(start_mat[:3, :3])
        rot_target = R.from_matrix(target_mat[:3, :3])
        slerp = Slerp([0, 1], R.from_quat(np.vstack([rot_start.as_quat(), rot_target.as_quat()])))
        rot_current = slerp([self.interp_progress])[0].as_matrix()

        self.current_R_mat[:3, :3] = rot_current
        self.current_R_mat[:3, 3] = pos_current

    def update(self):
        if not self.active:
            return None, None, None

        if self.state == "opening":
            if time.time() - self._opening_start_time >= HAND_OPEN_DURATION:
                self._begin_motion()
            # While opening, wrist holds at start_R_mat (current_R_mat unchanged).

        elif self.state in ("lifting", "approaching", "returning") and self.interp_progress < 1.0:
            now = time.time()
            if self._last_update_t is None:
                dt_real = self.dt
            else:
                # Clamp so a loop hiccup or state-transition gap can't snap
                # the wrist forward by more than ~one nominal step.
                dt_real = min(now - self._last_update_t, 0.1)
            self._last_update_t = now

            self.interp_progress = min(1.0, self.interp_progress + (self._active_interp_speed * dt_real))
            target_mat = self._lift_target if self.state == "lifting" else self.target_R_mat
            self._interpolate(self.start_R_mat, target_mat)

            if self.state == "lifting" and self.interp_progress >= 1.0:
                self._start_lateral_approach()

        return self.current_L_mat, self.current_R_mat, self.hand_state
