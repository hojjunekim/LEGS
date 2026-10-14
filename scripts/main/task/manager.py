"""Task manager — steps through a scene's declared `task_plan`."""

from .primitives import (
    Approach, Drive, Grasp, Heading, Perturb, Place, Pour, Stepback,
    VerifyHeld, VerifyPlaced, YawAlign, Status,
)

_PRIMITIVE_TYPES = {
    "approach":       Approach,
    "stepback":       Stepback,
    "heading":        Heading,
    "drive":          Drive,
    "yaw_align":      YawAlign,
    "grasp":          Grasp,
    "place":          Place,
    "pour":           Pour,
    "perturb":        Perturb,
    "verify_held":    VerifyHeld,
    "verify_placed":  VerifyPlaced,
}


class TaskManager:
    """Holds the plan + current primitive. Two run modes:

    - Manual: operator clicks "Next" → `start_next()` → primitive runs to
      SUCCESS, then the manager goes idle awaiting the next click.
    - Auto:   `run_all()` kicks off and each SUCCESS auto-advances until done.

    `reset()` (call on sim reset) returns to idx=0, idle.
    """

    def __init__(self, plan):
        self.plan = list(plan) if plan else []
        self.idx = 0
        self.current = None
        self.mode = "idle"   # "idle" | "running_one" | "running_all"
        self.failed_at = None  # idx where a primitive FAILED, or None

    # ---------------- state queries ----------------
    def is_done(self):
        return self.idx >= len(self.plan)

    def failed(self):
        """True if the plan stopped because a primitive returned FAILED."""
        return self.failed_at is not None

    def label(self):
        if not self.plan:
            return "(no plan)"
        if self.failed():
            name, arg = self.plan[self.failed_at]
            return f"[{self.failed_at + 1}/{len(self.plan)}] {name}({arg}) — FAILED"
        if self.is_done() and self.mode == "idle":
            return f"[{len(self.plan)}/{len(self.plan)}] — SUCCESS"
        i = min(self.idx, len(self.plan) - 1)
        name, arg = self.plan[i]
        tag = "RUNNING" if self.mode != "idle" and self.current is not None else "READY"
        return f"[{i + 1}/{len(self.plan)}] {name}({arg}) — {tag}"

    # ---------------- control ----------------------
    def reset(self, app=None):
        self.idx = 0
        self.current = None
        self.mode = "idle"
        self.failed_at = None
        if app is not None:
            self._restore_idle_gripper(app)

    def start_next(self, app):
        """Start the next pending primitive. No-op if already running or done."""
        if self.mode != "idle" or self.is_done():
            return
        self._start_current(app)
        self.mode = "running_one"

    def run_all(self, app):
        """Kick off the plan and auto-advance through all remaining primitives."""
        if self.is_done():
            return
        if self.mode == "idle":
            self._start_current(app)
        self.mode = "running_all"

    def tick(self, app):
        """Advance the current primitive one control step."""
        if self.mode == "idle" or self.current is None:
            return

        status = self.current.step(app)
        if status == Status.RUNNING:
            return
        if status == Status.FAILED:
            print(f"[TaskManager] primitive FAILED: {self.current.label()} — stopping plan")
            self._restore_idle_gripper(app)
            self.failed_at = self.idx
            self.current = None
            self.mode = "idle"
            return
        # SUCCESS
        self._on_success(app)
        self.idx += 1
        if self.is_done():
            self.current = None
            self.mode = "idle"
            print("[TaskManager] plan complete")
            return
        if self.mode == "running_all":
            self._start_current(app)
        else:
            self.current = None
            self.mode = "idle"

    # ---------------- internals ----------------
    def _start_current(self, app):
        name, arg = self.plan[self.idx]
        cls = _PRIMITIVE_TYPES.get(name)
        if cls is None:
            raise ValueError(f"Unknown primitive '{name}'")
        self.current = cls(arg)
        status = self.current.start(app)
        if status == Status.FAILED:
            print(f"[TaskManager] primitive FAILED in start(): {self.current.label()}")
            self._restore_idle_gripper(app)
            self.failed_at = self.idx
            self.current = None
            self.mode = "idle"

    def _on_success(self, app):
        """Track what the robot is holding so later primitives can reason about it."""
        if self.current is None:
            return
        if isinstance(self.current, Grasp):
            app._held_object_name = self.current.arg
        elif isinstance(self.current, Place):
            app._held_object_name = None

    def _restore_idle_gripper(self, app):
        """Open the gripper back to the ideal 'lego' pose so the next attempt
        (or sim reset) doesn't start with closed fingers. Called from the
        failure paths and from `reset(app)` so manual Next/RunAll clicks AND
        between-episode resets both recover the gripper."""
        app.left_hand_vals = app.gui.get_hand_joints("lego", is_left_hand=True)
        app.right_hand_vals = app.gui.get_hand_joints("lego", is_left_hand=False)
        if hasattr(app.gui, "l_hand_pose"):
            app.gui.l_hand_pose.value = "lego"
        if hasattr(app.gui, "r_hand_pose"):
            app.gui.r_hand_pose.value = "lego"
        if hasattr(app, "upper_controller"):
            app.upper_controller.hand_state = "lego"
        app._held_object_name = None
