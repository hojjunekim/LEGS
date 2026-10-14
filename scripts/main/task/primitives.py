"""Task primitives — Approach, Grasp, Place.

Each primitive is a small state machine with:
  start(app) -> str    # returns a Status; called once when entering
  step(app)  -> str    # called each control tick; returns a Status

Primitives read live poses from `app.ros_node.get_pose(name)` at start() and
(where appropriate) each step, so randomized scenes "just work" — no replanning.
"""

import math
import random
import time
import numpy as np
import trimesh.transformations as tft
from scipy.spatial.transform import Rotation as R

from scene_config import (
    find_object,
    find_entity,
    static_top_offset,
)
from control.upper_body_control import IDEAL_L_WRIST_POS, IDEAL_R_WRIST_POS


# ---- Tunables (global defaults; override per-entity later if needed) -----
STAND_OFF = 0.28                 # distance (m) robot stops short of entity center
APPROACH_ENTITY_TOL = 0.05       # m — extra slack on top of STAND_OFF: Approach
                                 # succeeds as soon as the robot is within
                                 # STAND_OFF + this of the entity, *regardless*
                                 # of whether the robot ends up exactly on the
                                 # initial robot→entity line. YawAlign right
                                 # after handles the final body rotation, so
                                 # being a little off-line is fine.
LOWER_SETTLE_SEC = 0.0           # wait after pelvis reaches target height
APPROACH_SETTLE_SEC = 0.2        # wrist-at-target settle before grab/release
GRAB_HOLD_SEC = 0.3              # wait after closing fingers — let the gripper
                                 # fully close before pelvis-raise/arm-return
                                 # so the object doesn't slip during the lift.
RELEASE_HOLD_SEC = 1.0           # wait after opening fingers (place) — long
                                 # enough that the hand finishes stretching
                                 # before the pelvis starts standing back up.
RAISE_SETTLE_SEC = 0.3           # wait after pelvis returns to default height
PELVIS_RATE = 0.15               # m/s — continuous pelvis height ramp (constant
                                 # speed, so larger drops take more time)
PELVIS_REACH_TOL = 0.005         # m — height-target reached threshold
NO_SQUAT_GAP = 0.10              # m — squat dead band. If the surface top (or
                                 # wrist target) is within this much of default
                                 # pelvis height, skip the squat and stay at
                                 # default. Sized so high_table (top 0.73, ~5
                                 # cm below default 0.78) doesn't trigger a dip;
                                 # only mid/low tables and the floor do.
PLACE_Z_CLEARANCE = 0.10          # held-object center Z above target surface
                                 # on place — high enough that the object
                                 # clears the surface with no contact; release
                                 # lets it drop safely. Used for table-resting
                                 # held objects only; floor-standing held
                                 # objects derive their place pose from natural
                                 # rest (see Place).
PLACE_NATURAL_HOVER_BUFFER = 0.05 # hover this much above the natural rest pose
                                 # for floor-standing held objects (e.g. tripod
                                 # on tile). Descend by the same amount lands
                                 # the object body origin exactly at natural
                                 # rest on the new surface.
PRE_GRASP_STANDOFF_X = 0.05       # robot-base −X offset for the pre-grasp pose.
                                 # Phase 1 reaches a pose this far BEHIND the
                                 # grasp pose (in the robot's base −X direction);
                                 # Phase 2 then reaches forward in +X onto the
                                 # object. Approaching head-on (instead of from
                                 # above) avoids the wrist colliding with the
                                 # table when the grasp pose sits near the
                                 # tabletop.
PRE_GRASP_STANDOFF_Y = 0.03      # robot-base ±Y offset for the pre-grasp pose.
                                 # Sign matches arm side: +Y (left) for left
                                 # arm, −Y (right) for right arm. Shifts the
                                 # standoff outboard so the wrist comes in
                                 # diagonally from the arm's natural side
                                 # instead of straight head-on.
PRE_GRASP_STANDOFF_Z = 0.07       # robot-base +Z offset for the pre-grasp pose.
                                 # Small lift keeps the fingers above the table
                                 # during the head-on horizontal segment so the
                                 # hand doesn't get pinned by friction; the
                                 # reach descends 3 cm diagonally onto the
                                 # grasp pose. Bump to 0.05+ if fingers still
                                 # clip the tabletop.
PRE_GRASP_REACH_SPEED = 0.5      # interp speed (fraction/s) for the forward
                                 # reach segment — 2× the default wrist speed
                                 # so the final reach is quick and the rubber
                                 # gripper doesn't dwell beside the object.
GRASP_PALM_Y_THRESHOLD = 0.07    # m — |object.y in palm frame| above which the
                                 # palm-y check counts as bad.
GRASP_PALM_Z_THRESHOLD = 0.02    # m — |object.z in palm frame| above which the
                                 # palm-z check counts as bad. Tighter than
                                 # the other axes because palm.z drift maps
                                 # ~directly to vertical wrist mis-alignment
                                 # against the table surface.
GRASP_THUMB_X_THRESHOLD = 0.08   # m — |object.x in thumb (link 2) frame| above
                                 # which the thumb-side check counts as bad.
                                 # Success requires |palm.y|, |palm.z|, AND
                                 # |thumb.x| all within their tolerances; any
                                 # axis above triggers a servo correction.
GRASP_SERVO_GAIN = 0.7            # fraction of palm.y / thumb.x error applied
                                 # as a wrist-target shift per servo step.
                                 # <1 to avoid overshoot but still converge in
                                 # a handful of iterations.
GRASP_SERVO_MAX_ITERATIONS = 2   # max servo corrections after a failed reach.
                                 # _apply_servo_correction branches on the
                                 # iteration count: 1st failure → no-lift
                                 # lateral nudge; 2nd failure → lift+descend
                                 # with corrected target. Exhaust →
                                 # _maybe_start_outer_retry.
GRASP_OUTER_RETRY_MAX = 1        # max full-restart retries after the servo
                                 # budget is exhausted. One outer retry sends
                                 # the arm to idle, runs a fresh YawAlign
                                 # (resamples a new random hand yaw offset),
                                 # then re-runs _begin_arm_approach. Total
                                 # worst-case attempts: 1 initial + 1 no-lift
                                 # adjust + 1 lift+descend + 1 outer retry ×
                                 # (1 initial + 1 no-lift + 1 lift+descend)
                                 # = 6 grasp attempts.
GRASP_SERVO_MAX_TOTAL_SHIFT = 0.10  # m — cumulative wrist-target shift cap
                                 # (measured from the planner's original
                                 # target) across all servo retries; abort if
                                 # exceeded so corrections can't drag the
                                 # wrist arbitrarily far from the planned pose.
GRASP_SERVO_LIFT_HEIGHT = 0.03   # m — on a failed grasp check, lift the wrist
                                 # straight up by this much from its actual
                                 # current pose, then descend to the laterally
                                 # corrected grasp pose. Starting the retry
                                 # from the air avoids the open hand dragging
                                 # across the table on the way in.
PRE_PLACE_DESCEND_SPEED = 1.5     # interp speed (fraction/s) for the descent
                                 # segment of Place — same default as the
                                 # grasp plunge; tune independently if needed.
YAW_ALIGN_HAND_OFFSET_DEG_RANGE = (10.0, 25.0)  # body yaw bias (deg) for
                                 # YawAlign — sampled uniformly per start()
                                 # so per-episode the hand-favored facing
                                 # varies, broadening the policy's exposure
                                 # to non-canonical pre-grasp headings. Right
                                 # hand → body turns left by that many deg
                                 # (target lands on robot's right); left hand
                                 # → mirror. Sampled ONCE at start() so the
                                 # target heading is constant across the
                                 # rotation; per-tick sampling would jitter
                                 # the target and prevent convergence.
YAW_ALIGN_SETTLE_TICKS = 3        # consecutive in-tolerance ticks required
                                 # before YawAlign returns SUCCESS. Guards
                                 # against firing during a momentary zero-
                                 # crossing while residual base rotation
                                 # (e.g. from a preceding Perturb) is still
                                 # carrying the heading through the target.
POUR_X_OFFSET = 0.02              # forward offset in robot local frame (m); pushes pour target away from robot
POUR_Y_OFFSET = 0.05              # lateral offset in robot local frame (m); +Y for left, -Y for right
POUR_Z_CLEARANCE = 0.12          # wrist Z above target surface on pour (small lift over mug rim)
POUR_TILT_DEG = 120               # wrist roll (local X) at pour apex
POUR_HOLD_SEC = 1.5              # dwell at full tilt

# Verify-primitive tolerances (world frame).
VERIFY_HELD_SETTLE_SEC = 0.1     # wait before sampling apple/wrist pose
VERIFY_HELD_Z_TOL = 0.1         # max |apple.z - wrist.z| when held
VERIFY_PLACED_SETTLE_SEC = 1.0   # wait for object to settle on surface
VERIFY_PLACED_XY_TOL = 0.1      # max XY distance from surface center
VERIFY_PLACED_Z_TOL = 0.06       # max (obj.z - surface.z); rest height of an
                                 # orange on the plate is ~0.03m, so 0.06m
                                 # admits all observed successes (max 0.047m)
                                 # while rejecting "still in hand" (~0.28m+).

# Step-back applied at the start of Approach when the robot is holding an
# object — gives clearance from the surface where the object was grabbed
# before walking forward to the next target.
STEPBACK_DURATION_SEC = 1.0
STEPBACK_VX = -0.25

class Status:
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _surface_top_world_z(app, surface_name):
    """World Z of the given entity's top-of-mesh, using its live pose.

    Works for both statics (tables) and objects (mug on a table).
    """
    cfg = find_entity(app.scene_cfg, surface_name)
    if cfg is None:
        return None
    live = app.ros_node.get_pose(surface_name)
    if live is None:
        return cfg["pos"][2] + static_top_offset(cfg)
    return float(live[0][2]) + static_top_offset(cfg)


def _pelvis_height_for_surface(app, surface_name):
    """Pelvis target = world Z of the surface top (no constant offset).

    If the surface top is already close to (or above) the default standing
    pelvis height, skip the squat — return default_height instead. Only
    surfaces meaningfully below default (mid/low tables) trigger a dip.
    """
    top = _surface_top_world_z(app, surface_name)
    default = app.controller.default_height
    if top is None:
        return default
    if top >= default - NO_SQUAT_GAP:
        return default
    return float(top)


def _wrist_world_xyz(app, use_left):
    """Current wrist position in world frame."""
    wrist_base = app.L_mat if use_left else app.R_mat
    return (app.T_base_world @ wrist_base)[:3, 3]


def _ramp_height(app, target_z, last_t):
    """Step app.controller.target_height toward target_z at PELVIS_RATE m/s.
    Returns (reached, now_t). Caller should pass now_t back as last_t next tick.
    """
    now = time.time()
    step = PELVIS_RATE * max(now - last_t, 0.0)
    cur = app.controller.target_height
    if cur > target_z:
        app.controller.target_height = max(target_z, cur - step)
    elif cur < target_z:
        app.controller.target_height = min(target_z, cur + step)
    reached = abs(app.controller.target_height - target_z) < PELVIS_REACH_TOL
    return reached, now


# ---------------------------------------------------------------------------
# Primitive base
# ---------------------------------------------------------------------------

class Primitive:
    kind = "primitive"

    def __init__(self, arg):
        self.arg = arg

    def label(self):
        return f"{self.kind}({self.arg})"

    def start(self, app):
        raise NotImplementedError

    def step(self, app):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Approach — drive lower body toward entity, stop at STAND_OFF facing it.
# ---------------------------------------------------------------------------

class Approach(Primitive):
    kind = "approach"

    def __init__(self, arg):
        super().__init__(arg)
        # `arg` may be a single entity name (str) or a 2-element list/tuple of
        # names. Two names → target is the XY midpoint of both poses, so the
        # robot drives to a point between them (useful for staging before a
        # subsequent yaw_align picks the specific grasp target).
        if isinstance(arg, (list, tuple)):
            self._target_names = list(arg)
            self._label = "+".join(str(a) for a in arg)
        else:
            self._target_names = [arg]
            self._label = str(arg)

    def _target_xy(self, app):
        """Live XY of the approach target. Midpoint when `arg` was a 2-list.
        Returns None if any constituent pose is unavailable."""
        poses = [app.ros_node.get_pose(n) for n in self._target_names]
        if any(p is None for p in poses):
            return None
        xs = [float(p[0][0]) for p in poses]
        ys = [float(p[0][1]) for p in poses]
        return np.array(
            [sum(xs) / len(xs), sum(ys) / len(ys)], dtype=np.float32
        )

    def label(self):
        return f"{self.kind}({self._label})"

    def start(self, app):
        # Approach always navigates at standing height. set_path used to do
        # this implicitly when called in start(); now that set_path is deferred
        # to _plan_and_drive (after yaw_align), reset the target up front so
        # stepback/yaw_align don't broadcast a stale low height left by a
        # prior Place/Pour and squat the robot mid-rotation.
        app.controller.target_height = app.controller.default_height

        if self._target_xy(app) is None:
            print(f"[Task] Approach '{self._label}': no pose yet, FAIL")
            return Status.FAILED

        # Path planning deferred until after yaw alignment so the path starts
        # from the robot's true pose. target_pos / direct_heading are computed
        # by _refresh_target — for the holding case we delay that until after
        # stepback so the standoff isn't 0.4m off-axis from the moved robot.
        self._prev_yaw = None
        self.target_pos = None
        self.direct_heading = None

        if getattr(app, "_held_object_name", None) is not None:
            self._state = "stepback"
            self._stepback_t0 = time.time()
            print(
                f"[Task] Approach '{self._label}' — stepping back "
                f"(holding '{app._held_object_name}') before yaw align"
            )
        else:
            if not self._refresh_target(app):
                return Status.FAILED
            self._state = "yaw_align"
            print(
                f"[Task] Approach '{self._label}' → yaw align toward "
                f"({self.target_pos[0]:.2f}, {self.target_pos[1]:.2f})"
            )
        return Status.RUNNING

    def _refresh_target(self, app):
        """Recompute target_pos + direct_heading from the live robot/entity poses.

        Called once before yaw_align: at start() for the no-hold path, and at
        the stepback→yaw_align transition for the holding path. Recomputing
        post-stepback keeps the standoff on the new robot→entity line
        (otherwise a 2 s backward step pushes the cached standoff to the side
        of the goal and the final direct_heading aims wrong).
        """
        xy = self._target_xy(app)
        if xy is None:
            print(f"[Task] Approach '{self._label}': no pose during refresh, FAIL")
            return False
        rx, ry = float(app.robot_pos[0]), float(app.robot_pos[1])
        ex, ey = float(xy[0]), float(xy[1])
        dx, dy = ex - rx, ey - ry
        dist = float(max(1e-4, np.hypot(dx, dy)))
        ux, uy = dx / dist, dy / dist
        stand_off = min(STAND_OFF, max(0.0, dist - 0.05))
        self.target_pos = np.array(
            [ex - ux * stand_off, ey - uy * stand_off, 0.0], dtype=np.float32
        )
        self.direct_heading = float(np.arctan2(dy, dx))
        return True

    def _plan_and_drive(self, app):
        """Plan path from the live robot pose and hand control to the controller."""
        path = app.planner.plan_path(app.robot_pos, self.target_pos)
        if path is None:
            print(f"[Task] Approach '{self._label}': planner returned None, FAIL")
            return Status.FAILED
        app.path = path
        app.controller.set_path(
            path, end_action_name="stand", direct_heading=self.direct_heading,
            reach_dist=0.1,
        )
        app.controller.reached_goal = False
        app.server.scene.add_spline_catmull_rom(
            "/planned_path", positions=path, color=(25, 220, 50), line_width=4.0
        )
        app.controller.active = True
        self._state = "drive"
        print(f"[Task] Approach '{self._label}' → driving to standoff")
        return Status.RUNNING

    def step(self, app):
        if self._state == "stepback":
            if time.time() - self._stepback_t0 < STEPBACK_DURATION_SEC:
                # Pure backward — no yaw correction during stepback so the body
                # cleanly clears the surface before yaw alignment kicks in.
                app.action_cmd = {
                    "vx": STEPBACK_VX, "vy": 0.0, "vyaw": 0.0,
                    "height": float(app.controller.target_height),
                }
                app.gui.update_vel_display(STEPBACK_VX, 0.0, 0.0, height=app.action_cmd["height"])
                return Status.RUNNING
            # Stepback done — recompute target_pos / direct_heading from the
            # post-stepback pose. Without this, the standoff is on the line
            # from the *pre*-stepback robot to the entity, so it lands several
            # cm off-axis next to the goal and direct_heading is rotated.
            if not self._refresh_target(app):
                return Status.FAILED
            self._state = "yaw_align"
            self._prev_yaw = None
            print(
                f"[Task] Approach '{self._label}' → yaw align after stepback toward "
                f"({self.target_pos[0]:.2f}, {self.target_pos[1]:.2f})"
            )

        if self._state == "yaw_align":
            # In-place rotation toward target_pos before path planning, so the
            # planned path starts from the robot's true (post-stepback) pose
            # and the controller's cruise begins already on-heading.
            ctrl = app.controller
            r_robot = R.from_quat([
                app.robot_wxyz[1], app.robot_wxyz[2],
                app.robot_wxyz[3], app.robot_wxyz[0],
            ])
            robot_yaw = float(r_robot.as_euler('xyz')[2])
            dx = float(self.target_pos[0] - app.robot_pos[0])
            dy = float(self.target_pos[1] - app.robot_pos[1])
            target_yaw = float(np.arctan2(dy, dx))
            yaw_err = (target_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi

            if self._prev_yaw is None:
                self._prev_yaw = robot_yaw
            yaw_rate = (
                ((robot_yaw - self._prev_yaw + np.pi) % (2 * np.pi) - np.pi)
                / max(ctrl.dt, 1e-3)
            )
            self._prev_yaw = robot_yaw

            if abs(yaw_err) < ctrl.initial_yaw_tol:
                return self._plan_and_drive(app)

            v_yaw = float(np.clip(
                ctrl.kp_angular * yaw_err - ctrl.kd_angular * yaw_rate,
                -ctrl.max_ang_vel, ctrl.max_ang_vel,
            ))
            app.action_cmd = {
                "vx": 0.0, "vy": 0.0, "vyaw": v_yaw,
                "height": float(ctrl.target_height),
            }
            app.gui.update_vel_display(0.0, 0.0, v_yaw, height=app.action_cmd["height"])
            return Status.RUNNING

        # state == "drive"
        t_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        vx, _vy, vyaw, height = app.controller.compute_command(
            app.robot_pos, app.robot_wxyz, self.target_pos, t_wxyz
        )
        # Walking phase: drop vy so the robot only drives forward with yaw
        # correction. vyaw stays on so the controller's reach test converges.
        app.action_cmd = {
            "vx": float(vx), "vy": 0.0,
            "vyaw": float(vyaw), "height": float(height),
        }
        app.gui.update_vel_display(vx, 0.0, vyaw, height=app.action_cmd["height"])

        # Proximity-based success: stop as soon as the robot is close enough
        # to the live entity pose. The controller's at_object_location flag
        # requires being on the precomputed initial→entity line *and* yaw-
        # aligned to the standoff heading, which over-constrains the approach
        # when the entity is laterally offset. YawAlign next handles facing.
        xy = self._target_xy(app)
        if xy is not None:
            dx_e = float(xy[0]) - float(app.robot_pos[0])
            dy_e = float(xy[1]) - float(app.robot_pos[1])
            if float(np.hypot(dx_e, dy_e)) < STAND_OFF + APPROACH_ENTITY_TOL:
                app.controller.active = False
                app.action_cmd = {
                    "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
                    "height": float(app.controller.target_height),
                }
                print(f"[Task] Approach '{self._label}' complete (proximity)")
                return Status.SUCCESS

        if app.controller.at_object_location:
            # Fallback: controller's line-based check, kept so a run that
            # finishes "the old way" still ends cleanly.
            app.controller.active = False
            app.action_cmd = {
                "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
                "height": float(app.controller.target_height),
            }
            print(f"[Task] Approach '{self._label}' complete")
            return Status.SUCCESS
        return Status.RUNNING


# ---------------------------------------------------------------------------
# Stepback / Heading / Drive — explicit splits of the legacy Approach state
# machine. Plans that want fine-grained control (e.g. inserting Stepback after
# verify_held instead of relying on Approach's held-object branch) can use
# these directly. Approach is kept untouched for backward compatibility.
#
# Motion equivalence with Approach:
#   - Approach (no held)                ≡ Heading(arg) → Drive(arg)
#   - Approach (holding something)      ≡ Stepback     → Heading(arg) → Drive(arg)
# ---------------------------------------------------------------------------

class Stepback(Primitive):
    """Back up from the current heading. Pelvis rises to default during the
    motion; no yaw correction (caller is responsible for facing first)."""
    kind = "stepback"

    def start(self, app):
        # Match Approach.start() — undo any leftover squat from a prior Place
        # so stepback runs at standing height.
        app.controller.target_height = app.controller.default_height
        self._t0 = time.time()
        print(f"[Task] Stepback — backing up for {STEPBACK_DURATION_SEC:.1f}s")
        return Status.RUNNING

    def step(self, app):
        ctrl = app.controller
        if time.time() - self._t0 < STEPBACK_DURATION_SEC:
            app.action_cmd = {
                "vx": STEPBACK_VX, "vy": 0.0, "vyaw": 0.0,
                "height": float(ctrl.target_height),
            }
            app.gui.update_vel_display(STEPBACK_VX, 0.0, 0.0, height=app.action_cmd["height"])
            return Status.RUNNING
        app.action_cmd = {
            "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
            "height": float(ctrl.target_height),
        }
        app.gui.update_vel_display(0.0, 0.0, 0.0, height=app.action_cmd["height"])
        print("[Task] Stepback complete")
        return Status.SUCCESS


class Heading(Primitive):
    """In-place yaw to face an entity, **no hand bias**. Pre-walk rotation.

    Same target_yaw as Approach's internal yaw_align substate (which faced
    the standoff direction — colinear with the entity from the robot's POV,
    so the angle is identical). Use YawAlign (not Heading) post-Drive when
    the precise hand-side bias matters for grasp/place.
    """
    kind = "heading"

    def start(self, app):
        # Match Approach.start() — clear any leftover squat target.
        app.controller.target_height = app.controller.default_height
        if app.ros_node.get_pose(self.arg) is None:
            print(f"[Task] Heading '{self.arg}': no pose yet, FAIL")
            return Status.FAILED
        self._prev_yaw = None
        print(f"[Task] Heading → '{self.arg}' (no hand bias)")
        return Status.RUNNING

    def step(self, app):
        ctrl = app.controller
        pose = app.ros_node.get_pose(self.arg)
        if pose is None:
            print(f"[Task] Heading '{self.arg}': pose lost mid-step, FAIL")
            return Status.FAILED

        sx, sy = float(pose[0][0]), float(pose[0][1])
        r_robot = R.from_quat([
            app.robot_wxyz[1], app.robot_wxyz[2],
            app.robot_wxyz[3], app.robot_wxyz[0],
        ])
        robot_yaw = float(r_robot.as_euler('xyz')[2])
        target_yaw = float(np.arctan2(sy - app.robot_pos[1], sx - app.robot_pos[0]))
        yaw_err = (target_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi

        if self._prev_yaw is None:
            self._prev_yaw = robot_yaw
        yaw_rate = (
            ((robot_yaw - self._prev_yaw + np.pi) % (2 * np.pi) - np.pi)
            / max(ctrl.dt, 1e-3)
        )
        self._prev_yaw = robot_yaw

        if abs(yaw_err) < ctrl.initial_yaw_tol:
            app.action_cmd = {
                "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
                "height": float(ctrl.target_height),
            }
            app.gui.update_vel_display(0.0, 0.0, 0.0, height=app.action_cmd["height"])
            print(f"[Task] Heading '{self.arg}' complete (err={yaw_err:.3f} rad)")
            return Status.SUCCESS

        v_yaw = float(np.clip(
            ctrl.kp_angular * yaw_err - ctrl.kd_angular * yaw_rate,
            -ctrl.max_ang_vel, ctrl.max_ang_vel,
        ))
        app.action_cmd = {
            "vx": 0.0, "vy": 0.0, "vyaw": v_yaw,
            "height": float(ctrl.target_height),
        }
        app.gui.update_vel_display(0.0, 0.0, v_yaw, height=app.action_cmd["height"])
        return Status.RUNNING


class Drive(Primitive):
    """Plan path from live robot pose → standoff near entity, walk, succeed
    on proximity to live entity. Verbatim copy of Approach's drive substate
    plus the standoff/_plan_and_drive setup that ran at its yaw→drive edge.
    """
    kind = "drive"

    def start(self, app):
        if app.ros_node.get_pose(self.arg) is None:
            print(f"[Task] Drive '{self.arg}': no pose yet, FAIL")
            return Status.FAILED
        if not self._refresh_target(app):
            return Status.FAILED
        path = app.planner.plan_path(app.robot_pos, self.target_pos)
        if path is None:
            print(f"[Task] Drive '{self.arg}': planner returned None, FAIL")
            return Status.FAILED
        app.path = path
        app.controller.set_path(
            path, end_action_name="stand", direct_heading=self.direct_heading,
            reach_dist=0.1,
        )
        app.controller.reached_goal = False
        app.server.scene.add_spline_catmull_rom(
            "/planned_path", positions=path, color=(25, 220, 50), line_width=4.0
        )
        app.controller.active = True
        print(f"[Task] Drive '{self.arg}' → driving to standoff")
        return Status.RUNNING

    def _refresh_target(self, app):
        """Same standoff formula as Approach._refresh_target."""
        pose = app.ros_node.get_pose(self.arg)
        if pose is None:
            print(f"[Task] Drive '{self.arg}': no pose during refresh, FAIL")
            return False
        entity = pose[0]
        rx, ry = float(app.robot_pos[0]), float(app.robot_pos[1])
        ex, ey = float(entity[0]), float(entity[1])
        dx, dy = ex - rx, ey - ry
        dist = float(max(1e-4, np.hypot(dx, dy)))
        ux, uy = dx / dist, dy / dist
        stand_off = min(STAND_OFF, max(0.0, dist - 0.05))
        self.target_pos = np.array(
            [ex - ux * stand_off, ey - uy * stand_off, 0.0], dtype=np.float32
        )
        self.direct_heading = float(np.arctan2(dy, dx))
        return True

    def step(self, app):
        t_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        vx, _vy, v_yaw, height = app.controller.compute_command(
            app.robot_pos, app.robot_wxyz, self.target_pos, t_wxyz
        )
        # Forward v_yaw verbatim. Drive leaves cruise yaw correction off, so
        # the controller turns only while aligning at the start and inside
        # reach_dist (VelocityController.compute_command). DART-style recovery
        # perturbations are injected by the explicit Perturb primitive in the
        # task DSL, not here.
        app.action_cmd = {
            "vx": float(vx), "vy": 0.0,
            "vyaw": float(v_yaw), "height": float(height),
        }
        app.gui.update_vel_display(vx, 0.0, v_yaw, height=app.action_cmd["height"])

        pose = app.ros_node.get_pose(self.arg)
        if pose is not None:
            dx_e = float(pose[0][0]) - float(app.robot_pos[0])
            dy_e = float(pose[0][1]) - float(app.robot_pos[1])
            if float(np.hypot(dx_e, dy_e)) < STAND_OFF + APPROACH_ENTITY_TOL:
                app.controller.active = False
                app.action_cmd = {
                    "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
                    "height": float(app.controller.target_height),
                }
                print(f"[Task] Drive '{self.arg}' complete (proximity)")
                return Status.SUCCESS

        if app.controller.at_object_location:
            app.controller.active = False
            app.action_cmd = {
                "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
                "height": float(app.controller.target_height),
            }
            print(f"[Task] Drive '{self.arg}' complete")
            return Status.SUCCESS
        return Status.RUNNING


# ---------------------------------------------------------------------------
# YawAlign — in-place rotate the body to face an entity's live XY position.
# Standalone primitive (not buried inside Approach/Place) so task plans can
# insert it explicitly before each manipulation step.
# ---------------------------------------------------------------------------

class YawAlign(Primitive):
    kind = "yaw_align"

    def start(self, app):
        if app.ros_node.get_pose(self.arg) is None:
            print(f"[Task] YawAlign '{self.arg}': no pose yet, FAIL")
            return Status.FAILED
        self._prev_yaw = None
        self._settle_count = 0
        # Lock the active-hand decision at start so the yaw bias doesn't flip
        # mid-rotation as the wrist-to-target distances change.
        self._use_left = self._resolve_active_hand(app)
        # Sample the hand-offset magnitude once per primitive instance —
        # holding it fixed for the entire rotation is what lets the PD loop
        # converge; resampling per-tick would move the target under it.
        self._yaw_bias_deg = random.uniform(*YAW_ALIGN_HAND_OFFSET_DEG_RANGE)
        side = "left" if self._use_left else "right"
        print(
            f"[Task] YawAlign → '{self.arg}' ({side} hand, "
            f"±{self._yaw_bias_deg:.1f}° bias sampled from "
            f"{YAW_ALIGN_HAND_OFFSET_DEG_RANGE})"
        )
        return Status.RUNNING

    def _resolve_active_hand(self, app):
        """True if the active hand is left, False if right.
        Priority: currently-held hand → scene_cfg.force_arm → distance heuristic."""
        if getattr(app, "_held_object_name", None) is not None:
            return getattr(app, "_grasp_side", "right") == "left"
        forced = getattr(app, "scene_cfg", {}).get("force_arm")
        if forced in ("left", "right"):
            return forced == "left"
        pose = app.ros_node.get_pose(self.arg)
        if pose is None:
            return False
        target_xyz = np.asarray(pose[0], dtype=np.float64)
        L_now = _wrist_world_xyz(app, use_left=True)
        R_now = _wrist_world_xyz(app, use_left=False)
        return float(np.linalg.norm(L_now - target_xyz)) < float(np.linalg.norm(R_now - target_xyz))

    def step(self, app):
        ctrl = app.controller
        pose = app.ros_node.get_pose(self.arg)
        if pose is None:
            print(f"[Task] YawAlign '{self.arg}': pose lost mid-step, FAIL")
            return Status.FAILED

        sx, sy = float(pose[0][0]), float(pose[0][1])
        r_robot = R.from_quat([
            app.robot_wxyz[1], app.robot_wxyz[2],
            app.robot_wxyz[3], app.robot_wxyz[0],
        ])
        robot_yaw = float(r_robot.as_euler('xyz')[2])
        target_yaw = float(np.arctan2(sy - app.robot_pos[1], sx - app.robot_pos[0]))
        # Hand-aware bias: right hand → +yaw (body turns left, target lands on
        # robot's right). Left hand → -yaw (mirror).
        yaw_bias = float(np.radians(self._yaw_bias_deg))
        target_yaw += -yaw_bias if self._use_left else yaw_bias
        yaw_err = (target_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi

        if self._prev_yaw is None:
            self._prev_yaw = robot_yaw
        yaw_rate = (
            ((robot_yaw - self._prev_yaw + np.pi) % (2 * np.pi) - np.pi)
            / max(ctrl.dt, 1e-3)
        )
        self._prev_yaw = robot_yaw

        if abs(yaw_err) < ctrl.initial_yaw_tol:
            # Require N consecutive in-tolerance ticks before SUCCESS. A
            # preceding Perturb leaves the base with non-zero angular momentum;
            # without this gate the heading can sweep through the target on
            # one tick (yaw_err small, but yaw_rate non-zero) and YawAlign
            # would fire spuriously, kicking off Grasp before the base parks.
            self._settle_count += 1
            app.action_cmd = {
                "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
                "height": float(ctrl.target_height),
            }
            app.gui.update_vel_display(0.0, 0.0, 0.0, height=app.action_cmd["height"])
            if self._settle_count >= YAW_ALIGN_SETTLE_TICKS:
                print(
                    f"[Task] YawAlign '{self.arg}' complete "
                    f"(err={yaw_err:.3f} rad, settled {self._settle_count} ticks)"
                )
                return Status.SUCCESS
            return Status.RUNNING
        # Out of tolerance — base is still rotating; reset the settle counter
        # so the next in-tolerance streak has to start from zero again.
        self._settle_count = 0

        v_yaw = float(np.clip(
            ctrl.kp_angular * yaw_err - ctrl.kd_angular * yaw_rate,
            -ctrl.max_ang_vel, ctrl.max_ang_vel,
        ))
        app.action_cmd = {
            "vx": 0.0, "vy": 0.0, "vyaw": v_yaw,
            "height": float(ctrl.target_height),
        }
        app.gui.update_vel_display(0.0, 0.0, v_yaw, height=app.action_cmd["height"])
        return Status.RUNNING


# ---------------------------------------------------------------------------
# _ManipulateBase — shared "lower → arm → trigger → raise → return" machine.
# Grasp and Place differ only in (a) wrist target and (b) hand action at apex.
# ---------------------------------------------------------------------------

class _ManipulateBase(Primitive):
    """Generic lower-body / upper-body sequence shared by Grasp and Place."""

    def start(self, app):
        surface = self._surface_name(app)
        self._pelvis_target = self._compute_pelvis_target(app, surface)
        # Don't snap target_height here — step() ramps it at PELVIS_RATE.
        self._ramp_last_t = time.time()
        # Zero base velocity while we manipulate; height follows the ramp.
        app.action_cmd = {
            "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
            "height": float(app.controller.target_height),
        }

        self._state = "lower"
        self._phase_t = time.time()
        self._settle_t = None
        self._grasp_side_local = None
        print(
            f"[Task] {self.label()} → lower pelvis "
            f"{app.controller.target_height:.2f} → {self._pelvis_target:.2f}"
        )
        return Status.RUNNING

    # --- subclass hooks ------------------------------------------------------
    def _compute_pelvis_target(self, app, surface_name):
        """Skip the squat when the wrist's grasp/place target sits within
        NO_SQUAT_GAP of the default pelvis height (the arm can reach without
        lowering the pelvis). Otherwise fall back to the surface-top heuristic
        so the pelvis dips toward the surface."""
        default = app.controller.default_height
        use_left = getattr(app, '_grasp_side', 'right') == 'left'
        T_target = self._wrist_target_world(app, use_left)
        if T_target is not None:
            wrist_target_z = float(T_target[2, 3])
            if wrist_target_z >= default - NO_SQUAT_GAP:
                return default
        return _pelvis_height_for_surface(app, surface_name)

    def _surface_name(self, app):
        raise NotImplementedError

    def _wrist_target_world(self, app, use_left):
        """4x4 world transform for the wrist target, or None."""
        raise NotImplementedError

    def _apex_action(self, app, use_left):
        """Called once at the top of the arm-approach motion."""
        raise NotImplementedError

    def _return_hand_state(self):
        """Hand state string during return-to-ideal."""
        return "grab"

    # --- shared step loop ---------------------------------------------------
    def step(self, app):
        # Hold base still at the current target height for the duration.
        app.action_cmd = {
            "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
            "height": float(app.controller.target_height),
        }
        # Broadcast current height so the GUI reflects the squat/raise ramp;
        # the sub-ticks below don't update the display themselves.
        app.gui.update_vel_display(0.0, 0.0, 0.0, height=app.action_cmd["height"])

        if self._state == "lower":
            return self._tick_lower(app)

        if self._state == "approach":
            return self._tick_approach(app)

        if self._state == "apex_hold":
            return self._tick_apex_hold(app)

        if self._state == "raise":
            return self._tick_raise(app)

        if self._state == "return":
            return self._tick_return(app)

        return Status.RUNNING

    def _pick_arm(self, app):
        """Use whichever arm's current wrist is closer (in world frame) to its
        own grasp target. Falls back to the only available side if the other's
        target is missing. Honors scene-level `force_arm` ("left"/"right") to
        pin a side regardless of distance — needed when spawn randomization
        can flip the closer-arm decision across episodes."""
        forced = getattr(app, "scene_cfg", {}).get("force_arm")
        if forced in ("left", "right"):
            use_left = forced == "left"
            T_target = self._wrist_target_world(app, use_left=use_left)
            if T_target is not None:
                return use_left, T_target

        T_R_world = self._wrist_target_world(app, use_left=False)
        T_L_world = self._wrist_target_world(app, use_left=True)
        if T_R_world is None and T_L_world is None:
            return None, None
        if T_L_world is None:
            return False, T_R_world
        if T_R_world is None:
            return True, T_L_world
        L_now = _wrist_world_xyz(app, use_left=True)
        R_now = _wrist_world_xyz(app, use_left=False)
        L_cost = float(np.linalg.norm(L_now - T_L_world[:3, 3]))
        R_cost = float(np.linalg.norm(R_now - T_R_world[:3, 3]))
        use_left = L_cost < R_cost
        T_target = T_L_world if use_left else T_R_world
        return use_left, T_target

    def _approach_hand_state(self):
        """Hand state during the arm approach phase."""
        return "stretch"

    def _begin_arm_approach(self, app):
        use_left, T_target_world = self._pick_arm(app)
        if use_left is None:
            print(f"[Task] {self.label()}: no wrist target available, FAIL")
            return Status.FAILED

        # Lock the target in WORLD frame. _tick_approach re-projects it into
        # the live base frame each tick so small pelvis drift during the
        # approach doesn't pull the wrist off the true world-frame goal.
        self._locked_target_world = T_target_world.copy()

        T_base_inv = np.linalg.inv(app.T_base_world)
        target_local = T_base_inv @ T_target_world
        current_mat = app.L_mat if use_left else app.R_mat

        app.upper_controller.start_approach(
            current_mat, target_local, direct=True,
            hand_state=self._approach_hand_state(),
        )
        self._grasp_side_local = "left" if use_left else "right"
        app._grasp_use_left = use_left
        app._grasp_side = self._grasp_side_local
        print(f"[Task] {self.label()} → {self._grasp_side_local} arm approach")
        self._state = "approach"
        self._settle_t = None
        return Status.RUNNING

    def _tick_lower(self, app):
        """Ramp pelvis down at PELVIS_RATE, then hold for LOWER_SETTLE_SEC."""
        reached, self._ramp_last_t = _ramp_height(
            app, self._pelvis_target, self._ramp_last_t
        )
        if not reached:
            self._settle_t = None
            return Status.RUNNING
        if self._settle_t is None:
            self._settle_t = time.time()
        if time.time() - self._settle_t < LOWER_SETTLE_SEC:
            return Status.RUNNING
        # Height settled — lock arm target and begin approach.
        return self._begin_arm_approach(app)

    def _tick_approach(self, app):
        # Reproject the world-frame target into the live base frame each tick.
        # _begin_arm_approach locked self._locked_target_world; pelvis drifts
        # slightly while the body settles, so a base-frame target fixed at
        # approach-start would land at the wrong world position.
        if getattr(self, "_locked_target_world", None) is not None:
            T_base_inv = np.linalg.inv(app.T_base_world)
            app.upper_controller.target_R_mat = T_base_inv @ self._locked_target_world
        use_left = self._grasp_side_local == "left"
        if app.upper_controller.interp_progress >= 1.0:
            if self._settle_t is None:
                self._settle_t = time.time()
            elif time.time() - self._settle_t >= APPROACH_SETTLE_SEC:
                self._apex_action(app, use_left=use_left)
                self._phase_t = time.time()
                self._state = "apex_hold"
                self._settle_t = None
        return Status.RUNNING

    def _tick_apex_hold(self, app):
        hold = self._apex_hold_sec()
        if time.time() - self._phase_t >= hold:
            self._ramp_last_t = time.time()
            self._settle_t = None
            self._state = "raise"
            print(f"[Task] {self.label()} → raising pelvis to default")
        return Status.RUNNING

    def _tick_raise(self, app):
        reached, self._ramp_last_t = _ramp_height(
            app, app.controller.default_height, self._ramp_last_t
        )
        if not reached:
            self._settle_t = None
            return Status.RUNNING
        if self._settle_t is None:
            self._settle_t = time.time()
        if time.time() - self._settle_t >= RAISE_SETTLE_SEC:
            use_left = self._grasp_side_local == "left"
            current_mat = app.L_mat if use_left else app.R_mat
            app.upper_controller.start_return_to_ideal(current_mat, use_left=use_left)
            app.upper_controller.hand_state = self._return_hand_state()
            self._state = "return"
        return Status.RUNNING

    def _tick_return(self, app):
        if app.upper_controller.interp_progress >= 1.0:
            use_left = self._grasp_side_local == "left"
            # Sync GUI handles + hand pose to whatever state we ended in.
            if use_left:
                app.gui.left_wrist_handle.position = app.L_mat[:3, 3]
                app.gui.left_wrist_handle.wxyz = tft.quaternion_from_matrix(app.L_mat)
                app.gui.l_hand_pose.value = self._return_hand_state()
            else:
                app.gui.right_wrist_handle.position = app.R_mat[:3, 3]
                app.gui.right_wrist_handle.wxyz = tft.quaternion_from_matrix(app.R_mat)
                app.gui.r_hand_pose.value = self._return_hand_state()
            app.upper_controller.reset()
            print(f"[Task] {self.label()} complete")
            return Status.SUCCESS
        return Status.RUNNING

    def _apex_hold_sec(self):
        return GRAB_HOLD_SEC


# ---------------------------------------------------------------------------
# Grasp — lower, close fingers on object, raise, return arm (still holding).
# ---------------------------------------------------------------------------

class Grasp(_ManipulateBase):
    kind = "grasp"
    # Lowest pelvis height we'll squat to — keeps locomotion_mode in standing
    # (the deploy switches to crouch mode at <=0.5 m, see ros_bridge.py).
    MIN_PELVIS_HEIGHT = 0.5
    # Leave the wrist this much above the grasp target after squat, so the
    # arm finishes with a short downward reach instead of starting exactly on
    # target — keeps the grasp motion within the IK's comfortable range.
    SQUAT_BUFFER_Z = 0.10

    def _approach_hand_state(self):
        # Hand stays in neutral "lego" pose during approach so the open fingers
        # don't snag the object on the way in; trigger_grab at apex closes them.
        return "lego"

    def _begin_arm_approach(self, app):
        """Two-segment motion for table objects: stand off behind the grasp
        pose in robot-base −X at the same height, then reach forward in +X —
        approaches the object head-on instead of plunging through the table.
        Floor-standing objects (no `resting_on`) skip the standoff and
        interpolate directly to the grasp pose: there's no surface to clear,
        and the pelvis squat already places the wrist close to the target."""
        use_left, T_target_world = self._pick_arm(app)
        if use_left is None:
            print(f"[Task] {self.label()}: no wrist target available, FAIL")
            return Status.FAILED

        on_surface = self._surface_name(app) is not None
        if on_surface:
            R_base = app.T_base_world[:3, :3]
            y_sign = 1.0 if use_left else -1.0
            standoff_offset = R_base @ np.array(
                [-PRE_GRASP_STANDOFF_X, y_sign * PRE_GRASP_STANDOFF_Y, PRE_GRASP_STANDOFF_Z],
                dtype=np.float32,
            )
            T_standoff_world = T_target_world.copy()
            T_standoff_world[:3, 3] = T_standoff_world[:3, 3] + standoff_offset
            self._approach_phase = "standoff"
        else:
            # No standoff/reach — single interp straight to the grasp pose.
            # Phase set to "reach" so _tick_approach skips the standoff→reach
            # swap and falls through to settle/apex once the interp finishes.
            T_standoff_world = T_target_world.copy()
            self._approach_phase = "reach"

        self._final_target_world = T_target_world.copy()
        self._locked_target_world = T_standoff_world
        # Snapshot the planner's target position so servo-shift caps measure
        # distance from the original grasp pose rather than from the previous
        # iteration (which would let the target drift unboundedly).
        self._servo_original_target_pos = T_target_world[:3, 3].copy()
        # Preserved across outer-retry restarts (getattr → 0 on first call).
        # _start_outer_retry bumps the count and re-enters this method via
        # YawAlign success in the outer_yaw_align branch.
        self._outer_retry_count = getattr(self, "_outer_retry_count", 0)

        T_base_inv = np.linalg.inv(app.T_base_world)
        target_local = T_base_inv @ T_standoff_world
        current_mat = app.L_mat if use_left else app.R_mat

        app.upper_controller.start_approach(
            current_mat, target_local, direct=True,
            hand_state=self._approach_hand_state(),
        )
        self._grasp_side_local = "left" if use_left else "right"
        app._grasp_use_left = use_left
        app._grasp_side = self._grasp_side_local
        if on_surface:
            print(
                f"[Task] {self.label()} → {self._grasp_side_local} arm standoff "
                f"({PRE_GRASP_STANDOFF_X*100:.0f} cm back, "
                f"{PRE_GRASP_STANDOFF_Y*100:.0f} cm outboard, "
                f"{PRE_GRASP_STANDOFF_Z*100:.0f} cm up)"
            )
        else:
            print(
                f"[Task] {self.label()} → {self._grasp_side_local} arm direct "
                f"to grasp pose (floor-standing, no standoff)"
            )
        self._state = "approach"
        self._settle_t = None
        return Status.RUNNING

    def _tick_approach(self, app):
        """Standoff segment first; on completion, swap target and reach
        forward to the final grasp pose. Settle/apex only after the reach
        finishes."""
        if getattr(self, "_locked_target_world", None) is not None:
            T_base_inv = np.linalg.inv(app.T_base_world)
            app.upper_controller.target_R_mat = T_base_inv @ self._locked_target_world
        use_left = self._grasp_side_local == "left"

        # Yaw-align sub-state — runs after an outer retry's idle reset.
        # The previous body yaw may have settled into a bad pose under
        # contact, so we resample a random per-hand bias and let the body
        # rotate underneath the arm (which is parked at the IDEAL idle
        # pose in base frame and rides with the body).
        if self._approach_phase == "outer_yaw_align":
            ya_status = self._inline_yaw_align.step(app)
            if ya_status == Status.FAILED:
                self._inline_yaw_align = None
                self._set_grasp_status(
                    app, "outer-retry yaw_align failed; aborting"
                )
                return Status.FAILED
            if ya_status == Status.SUCCESS:
                self._inline_yaw_align = None
                print(
                    f"[Task] {self.label()} → outer-retry yaw-aligned; "
                    f"restarting approach from scratch"
                )
                return self._begin_arm_approach(app)
            return Status.RUNNING

        # Mid-motion early grasp: while the wrist is moving during reach,
        # a no-lift lateral adjust, the servo lift, or the servo descend,
        # check every tick whether the alignment is already within tolerance.
        # If so, freeze the wrist at its current pose and trigger the grab
        # — don't waste the rest of the motion or the settle delay.
        # servo_lift is included because the lift itself can break a stuck
        # contact and reveal a graspable pose before the descend even starts.
        if (self._approach_phase in ("reach", "servo_adjust", "servo_lift", "servo_descend")
                and app.upper_controller.interp_progress < 1.0
                and self._grasp_position_ok(app, use_left=use_left)):
            cur = app.upper_controller.current_R_mat.copy()
            app.upper_controller.target_R_mat = cur
            app.upper_controller.start_R_mat = cur
            app.upper_controller.interp_progress = 1.0
            self._set_grasp_status(
                app, "early grasp — condition met mid-motion, stopping arm"
            )
            self._apex_action(app, use_left=use_left)
            self._phase_t = time.time()
            self._state = "apex_hold"
            self._settle_t = None
            return Status.RUNNING

        if app.upper_controller.interp_progress >= 1.0:
            if self._approach_phase == "standoff":
                # Standoff complete — swap target to the planner's raw
                # grasp pose and start the head-on reach. Off-axis errors
                # are handled by the servo retry loop after the reach,
                # not by an FK pre-correct here.
                self._locked_target_world = self._final_target_world
                T_base_inv = np.linalg.inv(app.T_base_world)
                app.upper_controller.continue_to(
                    T_base_inv @ self._final_target_world,
                    speed=PRE_GRASP_REACH_SPEED,
                )
                self._approach_phase = "reach"
                print(
                    f"[Task] {self.label()} → reach forward "
                    f"{PRE_GRASP_STANDOFF_X*100:.0f} cm @ {PRE_GRASP_REACH_SPEED:.1f}/s"
                )
                return Status.RUNNING

            if self._approach_phase == "servo_lift":
                # Phase A done — wrist is in the air, clear of any contact.
                # Descend directly to the corrected grasp pose. Body yaw
                # stays at whatever it was during the failed reach — the
                # post-lift yaw_align used to live here but recordings
                # showed the extra rotation under a hovering wrist was
                # disturbing the grasp without improving alignment. If a
                # body re-yaw is needed it happens later, in the outer
                # retry (which goes arm-to-idle first).
                self._locked_target_world = self._final_target_world
                T_base_inv = np.linalg.inv(app.T_base_world)
                app.upper_controller.continue_to(
                    T_base_inv @ self._final_target_world,
                    speed=PRE_GRASP_REACH_SPEED,
                )
                self._approach_phase = "servo_descend"
                print(
                    f"[Task] {self.label()} → post-lift; descend to corrected "
                    f"grasp pose @ {PRE_GRASP_REACH_SPEED:.1f}/s"
                )
                return Status.RUNNING

            if self._settle_t is None:
                self._settle_t = time.time()
            elif time.time() - self._settle_t >= APPROACH_SETTLE_SEC:
                # Position quality check (palm.y + thumb.x). Run BEFORE closing
                # the fingers. "ok" → close and continue; "servo" → nudge the
                # wrist target and keep reaching toward the corrected pose;
                # "fail" → servo budget exhausted, abort so the recorder doesn't
                # capture an empty grasp as a successful demo.
                verdict = self._check_grasp_position(app, use_left=use_left)
                if verdict == "ok":
                    self._apex_action(app, use_left=use_left)
                    self._phase_t = time.time()
                    self._state = "apex_hold"
                    self._settle_t = None
                elif verdict == "servo":
                    if not self._apply_servo_correction(app):
                        return self._maybe_start_outer_retry(
                            app, "servo correction vetoed (shift cap)"
                        )
                    self._settle_t = None
                else:  # "fail"
                    return self._maybe_start_outer_retry(
                        app, "servo budget exhausted"
                    )
        return Status.RUNNING

    def _grasp_position_ok(self, app, use_left):
        """Lightweight 'is grasp pose acceptable?' check used to early-exit a
        reach/servo_descend motion. Pure FK math + threshold compare, no GUI
        updates, prints, or servo-state caching. Returns True iff all three
        axes are within tolerance; False on any miss or missing FK/object."""
        wrist_fk = getattr(app.kinematics, "last_wrist_poses_base", {})
        side = "left" if use_left else "right"
        T_palm_base = wrist_fk.get(f"{side}_hand_palm_link")
        T_thumb_base = wrist_fk.get(f"{side}_hand_thumb_2_link")
        obj_pose = app.ros_node.get_pose(self.arg)
        if T_palm_base is None or T_thumb_base is None or obj_pose is None or app.T_base_world is None:
            return False
        obj_world_h = np.append(np.asarray(obj_pose[0], dtype=np.float64), 1.0)
        T_palm_world = app.T_base_world @ T_palm_base
        T_thumb_world = app.T_base_world @ T_thumb_base
        o_palm = np.linalg.inv(T_palm_world) @ obj_world_h
        o_thumb = np.linalg.inv(T_thumb_world) @ obj_world_h
        return (abs(float(o_palm[1])) <= GRASP_PALM_Y_THRESHOLD
                and abs(float(o_palm[2])) <= GRASP_PALM_Z_THRESHOLD
                and abs(float(o_thumb[0])) <= GRASP_THUMB_X_THRESHOLD)

    def _check_grasp_position(self, app, use_left):
        """Verdict on object alignment in palm + thumb frames. Returns one of:
            "ok"    — both axes within tolerance, close fingers
            "servo" — either axis out of tolerance, nudge wrist target and keep
                      reaching (caller invokes _apply_servo_correction)
            "fail"  — either axis out of tolerance, servo budget exhausted, abort
        Success requires |palm.y| AND |thumb.x| to both be within tolerance.
        On "servo", caches palm/thumb world transforms + signed errors on self
        so _apply_servo_correction can reuse them without re-running FK math.
        Updates the GUI grasp_status_display + persistent last-grasp widget."""
        wrist_fk = getattr(app.kinematics, "last_wrist_poses_base", {})
        side = "left" if use_left else "right"
        T_palm_base = wrist_fk.get(f"{side}_hand_palm_link")
        T_thumb_base = wrist_fk.get(f"{side}_hand_thumb_2_link")
        T_wrist_base = wrist_fk.get(f"{side}_wrist_yaw_link")
        obj_pose = app.ros_node.get_pose(self.arg)
        if T_palm_base is None or T_thumb_base is None or obj_pose is None or app.T_base_world is None:
            self._set_grasp_status(
                app,
                f"check skipped (no palm/thumb FK or '{self.arg}' pose) — proceeding",
            )
            return "ok"
        obj_world_h = np.append(np.asarray(obj_pose[0], dtype=np.float64), 1.0)
        T_palm_world = app.T_base_world @ T_palm_base
        T_thumb_world = app.T_base_world @ T_thumb_base
        o_palm = np.linalg.inv(T_palm_world) @ obj_world_h
        o_thumb = np.linalg.inv(T_thumb_world) @ obj_world_h
        palm_y = float(o_palm[1])
        palm_z = float(o_palm[2])
        thumb_x = float(o_thumb[0])
        palm_y_bad = abs(palm_y) > GRASP_PALM_Y_THRESHOLD
        palm_z_bad = abs(palm_z) > GRASP_PALM_Z_THRESHOLD
        thumb_bad = abs(thumb_x) > GRASP_THUMB_X_THRESHOLD
        ok = not (palm_y_bad or palm_z_bad or thumb_bad)
        if ok:
            self._set_grasp_status(
                app,
                f"OK — palm.y={palm_y:+.3f} palm.z={palm_z:+.3f} "
                f"thumb.x={thumb_x:+.3f} m "
                f"(|palm.y|≤{GRASP_PALM_Y_THRESHOLD:.2f}, "
                f"|palm.z|≤{GRASP_PALM_Z_THRESHOLD:.2f}, "
                f"|thumb.x|≤{GRASP_THUMB_X_THRESHOLD:.2f}); closing fingers",
            )
            self._set_last_grasp_result(app, "OK", palm_y, palm_z, thumb_x)
            return "ok"
        servo_count = getattr(self, "_servo_count", 0)
        reasons = []
        if palm_y_bad:
            reasons.append(f"|palm.y| exceeds {GRASP_PALM_Y_THRESHOLD:.2f} m")
        if palm_z_bad:
            reasons.append(f"|palm.z| exceeds {GRASP_PALM_Z_THRESHOLD:.2f} m")
        if thumb_bad:
            reasons.append(f"|thumb.x| exceeds {GRASP_THUMB_X_THRESHOLD:.2f} m")
        reason_str = " AND ".join(reasons) if len(reasons) > 1 else reasons[0]
        if servo_count >= GRASP_SERVO_MAX_ITERATIONS:
            self._set_grasp_status(
                app,
                f"FAIL — palm.y={palm_y:+.3f} palm.z={palm_z:+.3f} "
                f"thumb.x={thumb_x:+.3f} m "
                f"({reason_str}), servo budget exhausted "
                f"({servo_count}/{GRASP_SERVO_MAX_ITERATIONS}); aborting task",
            )
            self._set_last_grasp_result(app, "FAIL", palm_y, palm_z, thumb_x)
            return "fail"
        # Only the offending axes drive the correction; an axis already inside
        # tolerance shouldn't push the wrist sideways.
        self._servo_palm_y = palm_y if palm_y_bad else 0.0
        self._servo_palm_z = palm_z if palm_z_bad else 0.0
        self._servo_thumb_x = thumb_x if thumb_bad else 0.0
        self._servo_T_palm_world = T_palm_world
        self._servo_T_thumb_world = T_thumb_world
        # Cache actual wrist pose so _apply_servo_correction can compute the
        # Phase-A lift target from the wrist's TRUE position. When the wrist
        # is pinned in table contact the commanded pose drifts below the
        # actual; lifting from the commanded pose would leave the wrist on
        # the table.
        self._servo_actual_wrist_world = (
            (app.T_base_world @ T_wrist_base)[:3, 3].copy()
            if T_wrist_base is not None else None
        )
        self._set_grasp_status(
            app,
            f"SERVO ({servo_count + 1}/{GRASP_SERVO_MAX_ITERATIONS}) "
            f"— palm.y={palm_y:+.3f} palm.z={palm_z:+.3f} "
            f"thumb.x={thumb_x:+.3f} m, {reason_str}; "
            f"lift {GRASP_SERVO_LIFT_HEIGHT*100:.1f} cm then descend to corrected pose",
        )
        return "servo"

    def _set_grasp_status(self, app, text):
        handle = getattr(app.gui, "grasp_status_display", None)
        if handle is not None:
            handle.content = f"**Grasp check:** {text}"
        print(f"[Task] {self.label()} grasp check: {text}")

    def _set_last_grasp_result(self, app, tag, palm_y, palm_z, thumb_x):
        """Persistent widget — only updated when a grasp check resolves to
        OK or FAIL. Stays visible while servo corrections churn the live
        grasp_status_display so the operator can keep an eye on the offset
        that triggered the last verdict."""
        handle = getattr(app.gui, "last_grasp_result_display", None)
        if handle is not None:
            handle.content = (
                f"**Last grasp:** {tag} palm.y={palm_y:+.3f} "
                f"palm.z={palm_z:+.3f} thumb.x={thumb_x:+.3f} m"
            )

    def _apply_servo_correction(self, app):
        """On a failed grasp check, run an in-place retry:
          - First failure (_servo_count == 0): lateral nudge with no lift —
            shift _final_target_world by the cached error and continue the
            wrist straight to the new target. Cheaper and faster than the
            lift+descend; works when the wrist isn't pinned in contact.
          - Subsequent failures: in-place lift+descend
              "servo_lift"     — lift wrist GRASP_SERVO_LIFT_HEIGHT above
                                 its ACTUAL current pose.
              "servo_descend"  — descend to the laterally corrected pose.
            Lift breaks any table-contact pin so the lateral correction can
            actually move the wrist; body yaw stays put (re-yaw belongs in
            the outer retry, not under a hovering wrist).
        Returns True if a retry was kicked off; False if the cumulative
        lateral shift cap was exceeded."""
        if getattr(self, "_servo_count", 0) == 0:
            return self._do_in_place_lateral_adjust(app)
        return self._do_in_place_lift_descend(app)

    def _do_in_place_lift_descend(self, app):
        """Kick off the in-place close-pelvis retry by computing the lateral
        correction and lifting the wrist GRASP_SERVO_LIFT_HEIGHT above its
        ACTUAL current pose. Lifting from the ACTUAL (not commanded) wrist
        matters under contact: IK saturation makes the commanded pose drift
        below the actual, so "lift the commanded pose" doesn't move the
        wrist at all. Sets phase to "servo_lift"; once that completes
        _tick_approach descends straight to the corrected grasp pose
        (phase "servo_descend"). Returns True if applied; False if the
        cumulative lateral shift cap would be exceeded."""
        palm_y = self._servo_palm_y
        palm_z = self._servo_palm_z
        thumb_x = self._servo_thumb_x
        # Palm y/z axes and thumb x-axis in world frame — directions that,
        # when the wrist translates along them, reduce |palm.y|, |palm.z|
        # and |thumb.x| respectively.
        palm_y_axis_world = self._servo_T_palm_world[:3, 1]
        palm_z_axis_world = self._servo_T_palm_world[:3, 2]
        thumb_x_axis_world = self._servo_T_thumb_world[:3, 0]
        z_delta_world = GRASP_SERVO_GAIN * palm_z * palm_z_axis_world

        # Suppress downward z correction when the actual wrist is already
        # below (or below+gap) the commanded pose — meaning the table is
        # holding the wrist up and IK has saturated. Pushing the commanded
        # z lower just builds contact force without moving the wrist.
        actual_pos = getattr(self, "_servo_actual_wrist_world", None)
        if actual_pos is not None and z_delta_world[2] < 0.0:
            z_drift = float(actual_pos[2]) - float(self._final_target_world[2, 3])
            if z_drift > 0.01:
                print(
                    f"[Task] {self.label()} servo: skipping z correction — "
                    f"table contact (actual z above commanded by {z_drift*100:.1f} cm)"
                )
                z_delta_world = np.zeros(3)

        delta = (
            GRASP_SERVO_GAIN * palm_y * palm_y_axis_world
            + z_delta_world
            + GRASP_SERVO_GAIN * thumb_x * thumb_x_axis_world
        )

        # Phase B target — corrected grasp pose. Lateral shift accumulates
        # into _final_target_world across retries; total shift cap measures
        # distance from the planner's original target.
        new_pos = self._final_target_world[:3, 3] + delta
        original_pos = getattr(self, "_servo_original_target_pos", None)
        total_shift = (
            float(np.linalg.norm(new_pos - original_pos))
            if original_pos is not None else 0.0
        )
        if total_shift > GRASP_SERVO_MAX_TOTAL_SHIFT:
            self._set_grasp_status(
                app,
                f"servo aborted — cumulative shift {total_shift*100:.1f} cm "
                f"would exceed cap {GRASP_SERVO_MAX_TOTAL_SHIFT*100:.1f} cm",
            )
            return False
        self._servo_total_shift = total_shift
        self._final_target_world[:3, 3] = new_pos

        # Phase A target — lift straight up from the actual wrist XY.
        # Orientation matches _final_target_world so Phase B continues the
        # same slerp without a sudden rotation change at the A→B handoff.
        actual_pos = getattr(self, "_servo_actual_wrist_world", None)
        lift_target_world = self._final_target_world.copy()
        if actual_pos is not None:
            lift_target_world[0, 3] = float(actual_pos[0])
            lift_target_world[1, 3] = float(actual_pos[1])
            lift_target_world[2, 3] = float(actual_pos[2]) + GRASP_SERVO_LIFT_HEIGHT
        else:
            # No FK available — fall back to lifting the commanded pose. Less
            # reliable under contact, but better than aborting outright.
            lift_target_world[2, 3] = (
                self._final_target_world[2, 3] + GRASP_SERVO_LIFT_HEIGHT
            )

        self._locked_target_world = lift_target_world
        T_base_inv = np.linalg.inv(app.T_base_world)
        app.upper_controller.continue_to(
            T_base_inv @ lift_target_world,
            speed=PRE_GRASP_REACH_SPEED,
        )
        self._approach_phase = "servo_lift"
        self._servo_count = getattr(self, "_servo_count", 0) + 1
        delta_norm = float(np.linalg.norm(delta))
        print(
            f"[Task] {self.label()} servo {self._servo_count}/"
            f"{GRASP_SERVO_MAX_ITERATIONS}: correction delta={delta_norm*100:.2f} cm "
            f"(total shift {total_shift*100:.1f} cm); lifting "
            f"{GRASP_SERVO_LIFT_HEIGHT*100:.1f} cm from actual wrist, "
            f"then descend"
        )
        return True

    def _do_in_place_lateral_adjust(self, app):
        """First-failure recovery: nudge _final_target_world by the cached
        palm/thumb error and continue the wrist straight to the new target
        — no intermediate lift. Cheaper and faster than lift+descend; works
        well when the wrist isn't pinned in contact. Falls back to
        lift+descend on the next failure if this still isn't enough.
        Returns True if applied; False if the cumulative lateral shift cap
        would be exceeded."""
        palm_y = self._servo_palm_y
        palm_z = self._servo_palm_z
        thumb_x = self._servo_thumb_x
        palm_y_axis_world = self._servo_T_palm_world[:3, 1]
        palm_z_axis_world = self._servo_T_palm_world[:3, 2]
        thumb_x_axis_world = self._servo_T_thumb_world[:3, 0]
        z_delta_world = GRASP_SERVO_GAIN * palm_z * palm_z_axis_world

        # Same table-contact suppression as lift+descend: if the actual
        # wrist sits above the commanded target, IK is saturated and
        # pushing the commanded pose further down just builds contact
        # force without moving the wrist.
        actual_pos = getattr(self, "_servo_actual_wrist_world", None)
        if actual_pos is not None and z_delta_world[2] < 0.0:
            z_drift = float(actual_pos[2]) - float(self._final_target_world[2, 3])
            if z_drift > 0.01:
                print(
                    f"[Task] {self.label()} servo (no-lift): skipping z correction — "
                    f"table contact (actual z above commanded by {z_drift*100:.1f} cm)"
                )
                z_delta_world = np.zeros(3)

        delta = (
            GRASP_SERVO_GAIN * palm_y * palm_y_axis_world
            + z_delta_world
            + GRASP_SERVO_GAIN * thumb_x * thumb_x_axis_world
        )

        new_pos = self._final_target_world[:3, 3] + delta
        original_pos = getattr(self, "_servo_original_target_pos", None)
        total_shift = (
            float(np.linalg.norm(new_pos - original_pos))
            if original_pos is not None else 0.0
        )
        if total_shift > GRASP_SERVO_MAX_TOTAL_SHIFT:
            self._set_grasp_status(
                app,
                f"servo (no-lift) aborted — cumulative shift {total_shift*100:.1f} cm "
                f"would exceed cap {GRASP_SERVO_MAX_TOTAL_SHIFT*100:.1f} cm",
            )
            return False
        self._servo_total_shift = total_shift
        self._final_target_world[:3, 3] = new_pos

        self._locked_target_world = self._final_target_world
        T_base_inv = np.linalg.inv(app.T_base_world)
        app.upper_controller.continue_to(
            T_base_inv @ self._final_target_world,
            speed=PRE_GRASP_REACH_SPEED,
        )
        self._approach_phase = "servo_adjust"
        self._servo_count = getattr(self, "_servo_count", 0) + 1
        delta_norm = float(np.linalg.norm(delta))
        print(
            f"[Task] {self.label()} servo {self._servo_count}/"
            f"{GRASP_SERVO_MAX_ITERATIONS} (no-lift): correction delta={delta_norm*100:.2f} cm "
            f"(total shift {total_shift*100:.1f} cm); nudging laterally to corrected pose"
        )
        return True

    def _maybe_start_outer_retry(self, app, reason):
        """Last-resort retry after the in-place servo budget is exhausted (or
        a servo correction was vetoed by the shift cap). Sends the arm to
        idle, re-runs YawAlign (which resamples a fresh per-hand bias from
        YAW_ALIGN_HAND_OFFSET_DEG_RANGE on start), then restarts the whole
        approach from scratch via _begin_arm_approach. Capped at
        GRASP_OUTER_RETRY_MAX total restarts; budget exhaustion → FAILED.
        Returns Status to propagate from _tick_approach."""
        cur = getattr(self, "_outer_retry_count", 0)
        if cur >= GRASP_OUTER_RETRY_MAX:
            self._set_grasp_status(
                app,
                f"{reason}; outer-retry budget exhausted "
                f"({cur}/{GRASP_OUTER_RETRY_MAX}); FAIL",
            )
            return Status.FAILED
        self._outer_retry_count = cur + 1
        return self._start_outer_retry(app, reason)

    def _start_outer_retry(self, app, reason):
        """Kick off one outer retry: send wrist to IDEAL idle pose (base
        frame) so the hand clears any contact, reset servo bookkeeping so
        the fresh approach gets a clean budget, and spawn YawAlign. The
        "outer_yaw_align" branch in _tick_approach hands off to
        _begin_arm_approach on YawAlign success — that re-derives the
        target from the live object pose, so any drift while the failed
        attempt was running gets picked up automatically."""
        use_left = self._grasp_side_local == "left"
        idle_target_local = np.eye(4)
        idle_target_local[:3, 3] = (
            IDEAL_L_WRIST_POS if use_left else IDEAL_R_WRIST_POS
        )
        app.upper_controller.continue_to(
            idle_target_local,
            speed=PRE_GRASP_REACH_SPEED,
        )
        # Idle target rides with the body during yaw rotation; clear the
        # world-frame lock so _tick_approach doesn't override target_R_mat.
        self._locked_target_world = None
        # Fresh servo budget for the restarted approach.
        self._servo_count = 0
        self._servo_total_shift = 0.0

        self._inline_yaw_align = YawAlign(self.arg)
        status = self._inline_yaw_align.start(app)
        if status == Status.FAILED:
            self._inline_yaw_align = None
            self._set_grasp_status(
                app, "outer retry aborted — YawAlign.start() failed"
            )
            return Status.FAILED
        self._approach_phase = "outer_yaw_align"
        self._set_grasp_status(
            app,
            f"OUTER RETRY ({self._outer_retry_count}/{GRASP_OUTER_RETRY_MAX}) "
            f"— {reason}; arm to idle + re-yaw with fresh random offset, "
            f"then restart approach",
        )
        return Status.RUNNING

    def _surface_name(self, app):
        obj_cfg = find_object(app.scene_cfg, self.arg)
        if obj_cfg is None:
            return None
        return obj_cfg.get("resting_on")

    def _compute_pelvis_target(self, app, surface_name):
        # Surface-resting objects still use the surface-top heuristic.
        if surface_name is not None:
            return super()._compute_pelvis_target(app, surface_name)
        # Floor-standing objects (no `resting_on`): squat so the current wrist
        # ends up SQUAT_BUFFER_Z above the grasp target, leaving a small
        # downward reach for the arm motion at the end.
        obj_cfg = find_object(app.scene_cfg, self.arg)
        pose = app.ros_node.get_pose(self.arg)
        if obj_cfg is None or pose is None:
            return app.controller.default_height
        grasp_offset = obj_cfg.get("grasp", {}).get("offset", [0, 0, 0])
        # Offset.z in robot base frame ≈ world Z component for upright robot.
        target_wrist_z = float(pose[0][2]) + float(grasp_offset[2])
        current_wrist_z = float((app.T_base_world @ app.R_mat)[2, 3])
        delta_z = (target_wrist_z + self.SQUAT_BUFFER_Z) - current_wrist_z
        pelvis_target = app.controller.default_height + delta_z
        return max(pelvis_target, self.MIN_PELVIS_HEIGHT)

    def _wrist_target_world(self, app, use_left):
        # Compute from self.arg directly — the render pipeline's T_drw_world
        # tracks `primary_object`, which is `objects[0]` (often a destination
        # like the plate, not what the plan wants to grasp).
        obj_cfg = find_object(app.scene_cfg, self.arg)
        pose = app.ros_node.get_pose(self.arg)
        if obj_cfg is None or pose is None:
            return None
        obj_pos, _ = pose

        grasp_cfg = obj_cfg.get("grasp", {})
        # Live GUI override (tune offset at runtime); fall back to scene config.
        gui_offset = getattr(app.gui, "get_grasp_offset", lambda: None)()
        if gui_offset is not None:
            offset = np.asarray(gui_offset, dtype=np.float32)
        else:
            offset = np.asarray(
                grasp_cfg.get("offset", [-0.1, -0.03, 0.03]), dtype=np.float32
            )
        rpy = np.asarray(
            grasp_cfg.get("rotation_euler", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        if use_left:
            rot = R.from_euler("xyz", [-rpy[0], rpy[1], -rpy[2]], degrees=True).as_matrix()
            t = np.array([offset[0], -offset[1], offset[2] - 0.01], dtype=np.float32)
        else:
            rot = R.from_euler("xyz", rpy, degrees=True).as_matrix()
            t = offset

        # Apply offset + rotation in robot base frame (not object frame). For
        # rotationally symmetric objects (orange/plate), grasp direction must
        # stay consistent regardless of object yaw randomization.
        R_base = app.T_base_world[:3, :3].astype(np.float32)
        T_target = np.eye(4, dtype=np.float32)
        T_target[:3, :3] = R_base @ rot
        T_target[:3, 3] = np.asarray(obj_pos, dtype=np.float32) + R_base @ t
        return T_target

    def _apex_action(self, app, use_left):
        app.upper_controller.trigger_grab()
        # Remember which object is now in the gripper so later Place primitives
        # can compute wrist→object offset and center the object on the target.
        app._grasp_object = self.arg

    def _return_hand_state(self):
        return "grab"

    def _apex_hold_sec(self):
        return GRAB_HOLD_SEC


# ---------------------------------------------------------------------------
# Place — lower, open fingers above target surface, raise, return arm (open).
# ---------------------------------------------------------------------------

class Place(_ManipulateBase):
    kind = "place"

    def _surface_name(self, app):
        return self.arg

    def _compute_pelvis_target(self, app, surface_name):
        """Place uses the surface top (plate top) directly — not the wrist
        target, which sits PLACE_Z_CLEARANCE + held-object offset above the
        surface and would otherwise pass the no-squat dead band even when the
        actual placement surface is well below default pelvis height."""
        return _pelvis_height_for_surface(app, surface_name)

    def _pick_arm(self, app):
        """Reuse whichever arm is holding the object."""
        use_left = getattr(app, '_grasp_side', 'right') == 'left'
        T_target = self._wrist_target_world(app, use_left)
        if T_target is None:
            return None, None
        return use_left, T_target

    def _approach_hand_state(self):
        return "grab"

    def _begin_arm_approach(self, app):
        """Single-segment motion: arrive at the hover pose (`place_z_clearance`
        above the surface — what `_wrist_target_world` already returns) and
        release. The held object drops onto the surface from that height,
        which keeps the wrist clear of the table during placement.

        `_tick_approach` then waits for the approach interp to finish and
        runs the settle/apex (finger-release) step."""
        use_left, T_hover_world = self._pick_arm(app)
        if use_left is None:
            print(f"[Task] {self.label()}: no wrist target available, FAIL")
            return Status.FAILED

        self._locked_target_world = T_hover_world.copy()

        T_base_inv = np.linalg.inv(app.T_base_world)
        target_local = T_base_inv @ T_hover_world
        current_mat = app.L_mat if use_left else app.R_mat

        app.upper_controller.start_approach(
            current_mat, target_local, direct=True,
            hand_state=self._approach_hand_state(),
        )
        self._grasp_side_local = "left" if use_left else "right"
        app._grasp_use_left = use_left
        app._grasp_side = self._grasp_side_local
        print(
            f"[Task] {self.label()} → {self._grasp_side_local} arm hover "
            f"(release from above surface, no descent)"
        )
        self._state = "approach"
        self._settle_t = None
        return Status.RUNNING

    def _tick_approach(self, app):
        """Hover segment first; on completion, swap target and descend
        straight down. Apex (open fingers) only after the descent finishes."""
        if getattr(self, "_locked_target_world", None) is not None:
            T_base_inv = np.linalg.inv(app.T_base_world)
            app.upper_controller.target_R_mat = T_base_inv @ self._locked_target_world
        use_left = self._grasp_side_local == "left"

        if app.upper_controller.interp_progress >= 1.0:
            if self._settle_t is None:
                self._settle_t = time.time()
            elif time.time() - self._settle_t >= APPROACH_SETTLE_SEC:
                self._apex_action(app, use_left=use_left)
                self._phase_t = time.time()
                self._state = "apex_hold"
                self._settle_t = None
        return Status.RUNNING

    def _held_object_cfg(self, app):
        """Scene config of the currently held object, or None."""
        held_name = getattr(app, "_grasp_object", None) or \
            getattr(app, "_held_object_name", None)
        if held_name is None:
            return None
        return find_object(app.scene_cfg, held_name)

    def _held_is_floor_standing(self, app):
        """True when the held object was originally floor-standing (no
        `resting_on`). Place mirrors Grasp for these: the hover pose lands
        the object body origin at its natural rest height above the new
        surface (a clean reverse of picking it up off the floor)."""
        cfg = self._held_object_cfg(app)
        return cfg is not None and not cfg.get("resting_on")

    def _wrist_target_world(self, app, use_left):
        surface_pose = app.ros_node.get_pose(self.arg)
        surface_cfg = find_entity(app.scene_cfg, self.arg)
        if surface_pose is None or surface_cfg is None:
            return None
        sx, sy, sz = (float(c) for c in surface_pose[0])
        top_z = sz + static_top_offset(surface_cfg)

        # Floor-standing held: object's natural rest height above the new
        # surface = original_pos.z (its z when floor-standing, assuming
        # floor at z=0). Hover with a small buffer so the descend lands
        # the object exactly on the surface — Place is a clean reverse of
        # Grasp, with only the surface_top added to the wrist Z.
        # Otherwise (table-resting held): use the legacy clearance knob.
        held_cfg = self._held_object_cfg(app)
        if held_cfg is not None and not held_cfg.get("resting_on"):
            natural_z = float(held_cfg["pos"][2])
            object_target_z = top_z + natural_z + PLACE_NATURAL_HOVER_BUFFER
        else:
            clearance = float(surface_cfg.get("place_z_clearance", PLACE_Z_CLEARANCE))
            object_target_z = top_z + clearance

        # Keep the wrist's current world-frame orientation.
        current_base = app.L_mat if use_left else app.R_mat
        T_world = app.T_base_world @ current_base

        # Target where the HELD OBJECT's center should land — dish center XY
        # at the chosen Z. The wrist then offsets by the fixed wrist→object
        # vector so the object (not the wrist) lands on center.
        object_target = np.array(
            [sx, sy, object_target_z], dtype=np.float64
        )
        obj_name = getattr(app, "_grasp_object", None)
        offset = np.zeros(3, dtype=np.float64)
        if obj_name is not None:
            obj_pose = app.ros_node.get_pose(obj_name)
            if obj_pose is not None:
                obj_world = np.asarray(obj_pose[0], dtype=np.float64)
                offset = T_world[:3, 3].astype(np.float64) - obj_world

        T_world[:3, 3] = object_target + offset
        return T_world

    def _apex_action(self, app, use_left):
        app.upper_controller.state = "grabbing"
        app.upper_controller.hand_state = "lego"

    def _return_hand_state(self):
        # Resting pose is open hand; arm returns to ideal with gripper open.
        return "lego"

    def _apex_hold_sec(self):
        return RELEASE_HOLD_SEC

    # Fall through to the default _ManipulateBase._tick_apex_hold, which
    # raises the pelvis to default after RELEASE_HOLD_SEC. Recording stops
    # only after the full primitive (raise + arm return) completes, so the
    # episode captures the robot standing back up.


# ---------------------------------------------------------------------------
# Pour — lower, wrist above target, tilt wrist (still grabbing), un-tilt, raise.
# Reuses whichever arm is holding the object (set by a prior Grasp).
# ---------------------------------------------------------------------------

class Pour(_ManipulateBase):
    kind = "pour"

    def _surface_name(self, app):
        return self.arg

    def _pick_arm(self, app):
        use_left = getattr(app, '_grasp_side', 'right') == 'left'
        T_target = self._wrist_target_world(app, use_left)
        if T_target is None:
            return None, None
        return use_left, T_target

    def _approach_hand_state(self):
        return "grab"

    def _return_hand_state(self):
        return "grab"

    def _wrist_target_world(self, app, use_left):
        surface_pose = app.ros_node.get_pose(self.arg)
        surface_cfg = find_entity(app.scene_cfg, self.arg)
        if surface_pose is None or surface_cfg is None:
            return None
        sx, sy, sz = (float(c) for c in surface_pose[0])
        top_z = sz + static_top_offset(surface_cfg)

        current_base = app.L_mat if use_left else app.R_mat
        T_world = app.T_base_world @ current_base

        # Lateral offset in robot local frame: left hand pours from +Y, right from -Y.
        # +X pushes the pour target forward (away from robot) so it lands on mug center.
        y_local = POUR_Y_OFFSET if use_left else -POUR_Y_OFFSET
        local_offset = np.array([POUR_X_OFFSET, y_local, 0.0], dtype=np.float64)
        world_offset = app.T_base_world[:3, :3] @ local_offset

        # Desired bottle center: above target surface, laterally offset
        bottle_target = np.array(
            [sx + world_offset[0], sy + world_offset[1], top_z + POUR_Z_CLEARANCE],
            dtype=np.float64,
        )

        # Backtrack from desired bottle position to wrist position using
        # the live wrist→object vector (accounts for grasp offset + any drift).
        obj_name = getattr(app, "_grasp_object", None)
        wrist_to_obj = np.zeros(3, dtype=np.float64)
        if obj_name is not None:
            obj_pose = app.ros_node.get_pose(obj_name)
            if obj_pose is not None:
                obj_world = np.asarray(obj_pose[0], dtype=np.float64)
                wrist_to_obj = obj_world - T_world[:3, 3].astype(np.float64)

        # wrist_target = bottle_target - wrist_to_obj
        T_world[:3, 3] = bottle_target - wrist_to_obj
        return T_world

    def _apex_action(self, app, use_left):
        self._pour_phase = "tilt"
        self._pour_phase_t = time.time()
        self._start_tilt(app, use_left, degrees=POUR_TILT_DEG)

    def _start_tilt(self, app, use_left, degrees):
        # Roll the wrist about its LOCAL X axis. Arms roll with opposite signs
        # so both palms tip outward (thumb-down) for a natural pour.
        current_mat = app.L_mat if use_left else app.R_mat
        sign = 1.0 if use_left else -1.0
        roll = R.from_euler('x', sign * degrees, degrees=True).as_matrix()
        tilted = current_mat.copy()
        tilted[:3, :3] = current_mat[:3, :3] @ roll
        app.upper_controller.start_approach(
            current_mat, tilted, direct=True, hand_state="grab"
        )

    def _tick_apex_hold(self, app):
        use_left = self._grasp_side_local == "left"

        if self._pour_phase == "tilt":
            if app.upper_controller.interp_progress >= 1.0:
                self._pour_phase = "hold"
                self._pour_phase_t = time.time()
                print(f"[Task] {self.label()} pouring (hold {POUR_HOLD_SEC:.1f}s)")
            return Status.RUNNING

        if self._pour_phase == "hold":
            if time.time() - self._pour_phase_t >= POUR_HOLD_SEC:
                self._pour_phase = "untilt"
                self._start_tilt(app, use_left, degrees=-POUR_TILT_DEG)
            return Status.RUNNING

        if self._pour_phase == "untilt":
            if app.upper_controller.interp_progress >= 1.0:
                # Skip the pelvis raise: after pouring, retract the arm to
                # ideal while the pelvis stays low. The episode-end reset
                # restores default height, so raising here would only bounce
                # the body up just to snap back down on the next reset.
                current_mat = app.L_mat if use_left else app.R_mat
                app.upper_controller.start_return_to_ideal(current_mat, use_left=use_left)
                app.upper_controller.hand_state = self._return_hand_state()
                self._state = "return"
                print(f"[Task] {self.label()} → arm return (pelvis stays at {app.controller.target_height:.2f})")
            return Status.RUNNING

        return Status.RUNNING


# ---------------------------------------------------------------------------
# VerifyHeld / VerifyPlaced — single-shot physics checks between subtasks.
# Each settles briefly, then samples world poses and returns SUCCESS/FAILED.
# Used by auto-record to discard failed episodes.
# ---------------------------------------------------------------------------

class VerifyHeld(Primitive):
    """Confirm the grasped object's Z is close to the active wrist's Z."""
    kind = "verify_held"

    def start(self, app):
        self._t_start = time.time()
        return Status.RUNNING

    def step(self, app):
        if time.time() - self._t_start < VERIFY_HELD_SETTLE_SEC:
            return Status.RUNNING
        pose = app.ros_node.get_pose(self.arg)
        if pose is None:
            print(f"[Task] {self.label()}: no object pose, FAIL")
            return Status.FAILED
        obj_z = float(pose[0][2])
        use_left = getattr(app, "_grasp_side", "right") == "left"
        wrist_z = float(_wrist_world_xyz(app, use_left)[2])
        dz = abs(obj_z - wrist_z)
        ok = dz <= VERIFY_HELD_Z_TOL
        tag = "SUCCESS" if ok else "FAIL"
        print(
            f"[Task] {self.label()}: |obj.z - wrist.z| = {dz:.3f} m "
            f"(tol {VERIFY_HELD_Z_TOL:.2f}) → {tag}"
        )
        return Status.SUCCESS if ok else Status.FAILED


class VerifyPlaced(Primitive):
    """Confirm the most-recently-grasped object's XY is near the surface center."""
    kind = "verify_placed"

    def start(self, app):
        self._t_start = time.time()
        return Status.RUNNING

    def step(self, app):
        if time.time() - self._t_start < VERIFY_PLACED_SETTLE_SEC:
            return Status.RUNNING
        obj_name = getattr(app, "_grasp_object", None)
        if obj_name is None:
            print(f"[Task] {self.label()}: no grasped-object memory, FAIL")
            return Status.FAILED
        obj_pose = app.ros_node.get_pose(obj_name)
        surf_pose = app.ros_node.get_pose(self.arg)
        if obj_pose is None or surf_pose is None:
            print(f"[Task] {self.label()}: missing pose for '{obj_name}' / '{self.arg}', FAIL")
            return Status.FAILED
        ox, oy, oz = float(obj_pose[0][0]), float(obj_pose[0][1]), float(obj_pose[0][2])
        sx, sy, sz = float(surf_pose[0][0]), float(surf_pose[0][1]), float(surf_pose[0][2])
        dxy = float(np.hypot(ox - sx, oy - sy))
        dz = oz - sz
        ok_xy = dxy <= VERIFY_PLACED_XY_TOL
        ok_z = dz <= VERIFY_PLACED_Z_TOL
        ok = ok_xy and ok_z
        tag = "SUCCESS" if ok else "FAIL"
        print(
            f"[Task] {self.label()}: |'{obj_name}' - '{self.arg}'|_xy = {dxy:.3f} m "
            f"(tol {VERIFY_PLACED_XY_TOL:.2f}), dz = {dz:+.3f} m "
            f"(tol {VERIFY_PLACED_Z_TOL:.2f}) → {tag}"
        )
        return Status.SUCCESS if ok else Status.FAILED


# ---------------------------------------------------------------------------
# Perturb — boundary-state DART-style noise injector.
# Sits between primitives (e.g. after Stepback or after YawAlign) and emits a
# short signed vyaw burst (optionally with lateral vy) so the recorded actions
# include "robot ends primitive X slightly off-canonical, here's the next
# action conditioned on that state". Used to expand the policy's distribution
# of primitive-exit states beyond what the expert demonstrator naturally hits.
# ---------------------------------------------------------------------------

class Perturb(Primitive):
    """Apply a short randomized base-velocity perturbation.

    arg is a dict:
        yaw_deg:  half-range (deg) of the yaw bias applied over `duration`;
                  sign is sampled uniformly. vyaw = ±yaw_deg/duration.
        duration: window length (seconds).
        lateral:  optional half-range (m/s) of lateral vy; sign sampled
                  uniformly. Default 0 (yaw only).
    """
    kind = "perturb"

    def start(self, app):
        cfg = self.arg or {}
        yaw_half = float(cfg.get("yaw_deg", 0.0))
        self._duration = float(cfg.get("duration", 0.3))
        lat_half = float(cfg.get("lateral", 0.0))

        # Sample magnitude AND sign — uniform over [-yaw_half, +yaw_half]
        # so the total yaw kick (and lateral nudge) varies episode-to-episode.
        yaw_kick_deg = random.uniform(-yaw_half, yaw_half)
        self._vyaw = math.radians(yaw_kick_deg) / max(self._duration, 1e-3)
        self._vy = random.uniform(-lat_half, lat_half) if lat_half > 0.0 else 0.0

        self._t0 = time.time()
        # State-only: drive the physics with the perturb command, but skip
        # recording so the high vyaw never enters the labels. The following
        # primitive (typically YawAlign) provides the recorded recovery action.
        app._skip_record = True
        print(
            f"[Task] Perturb (kick={yaw_kick_deg:+.1f}°, vyaw={math.degrees(self._vyaw):+.1f}°/s, "
            f"vy={self._vy:+.2f} m/s, {self._duration:.2f}s) — state-only"
        )
        return Status.RUNNING

    def step(self, app):
        ctrl = app.controller
        elapsed = time.time() - self._t0
        if elapsed < self._duration:
            app.action_cmd = {
                "vx": 0.0, "vy": float(self._vy), "vyaw": float(self._vyaw),
                "height": float(ctrl.target_height),
            }
            app.gui.update_vel_display(0.0, self._vy, self._vyaw, height=app.action_cmd["height"])
            return Status.RUNNING
        app.action_cmd = {
            "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
            "height": float(ctrl.target_height),
        }
        app.gui.update_vel_display(0.0, 0.0, 0.0, height=app.action_cmd["height"])
        app._skip_record = False
        print("[Task] Perturb complete")
        return Status.SUCCESS
