"""Policy inference mixin for VisualizerApp.

Handles policy client connection, observation building, action application,
and both sequential and RTC execution modes.
"""

import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh.transformations as tft

from utils.psi0_client import BACKENDS, make_client

# Gripper command in the LEGS data: 0.5 = open ("lego" hand), 1.0 = closed.
GRIP_OPEN = 0.5


def mat4_to_6d(T: np.ndarray) -> np.ndarray:
    pos = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler('xyz')
    return np.concatenate([pos, rpy]).astype(np.float32)


def _fmt_action(a):
    if a is None:
        return "None"
    a = np.asarray(a)
    if a.size < 14:
        return f"<short action, size={a.size}>"
    s = (f"L_pos=[{a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}] "
         f"L_grip={a[6]:.2f} "
         f"R_pos=[{a[7]:+.3f}, {a[8]:+.3f}, {a[9]:+.3f}] "
         f"R_grip={a[13]:.2f}")
    if a.size >= 18:
        s += f" vx={a[14]:+.3f} vy={a[15]:+.3f} vyaw={a[16]:+.3f} h={a[17]:.3f}"
    elif a.size >= 15:
        s += f" h={a[14]:.3f}"
    return s


class PolicyRunnerMixin:
    """Mixin providing policy inference and action execution."""

    def _init_policy_state(self):
        """Initialize policy-related state variables."""
        if not hasattr(self, 'policy'):
            self.policy = None
        self._rtc_chunks = None
        self._rtc_chunk_list = []
        self._rtc_tick = 0
        self._rtc_lock = threading.Lock()
        self._rtc_inflight = False
        self._policy_was_disabled = True
        self._last_applied_action = None

    def _connect_policy(self):
        """Connect to the policy server named by --backend / --policy_host."""
        if not self.args.policy_host:
            return
        self.policy_backend = getattr(self.args, 'backend', None) or "psi0"
        spec = BACKENDS[self.policy_backend]
        host = self.args.policy_host
        port = getattr(self.args, 'policy_port', None) or spec["port"]
        # The scene's prompt is the one the policy was trained on.
        scene_cfg = getattr(self, 'scene_cfg', None) or {}
        self.policy_prompt = getattr(self.args, 'policy_prompt', None) or scene_cfg.get("prompt", "")
        self.policy_chunk = spec["chunk"]
        print(f"[Policy] Connecting to {self.policy_backend} server at {host}:{port} "
              f"(prompt: '{self.policy_prompt}')...")
        self.policy = make_client(self.policy_backend, host=host, port=port)
        # Clients stay alive when the server is down (so the sim can run while
        # it boots) and report status via `connected`.
        if self.policy.connected:
            self.gui.policy_status.content = "Policy: **connected** (paused)"
            print(f"[Policy] Connected! (backend={self.policy_backend}, chunk={self.policy_chunk})")
        else:
            self.gui.policy_status.content = "Policy: **server unreachable**"
            print(f"[Policy] Server unreachable at {host}:{port} — "
                  f"start the {self.policy_backend} server then toggle the policy.")

    def _policy_state(self):
        """18-D state in the observation.state_psi0_18 layout:
        L_wrist xyz+rpy (6), L_grip, R_wrist xyz+rpy (6), R_grip,
        previous base command vx, vy, vyaw, and height.
        """
        left_6d = mat4_to_6d(self.L_mat) if self.L_mat is not None else np.zeros(6, dtype=np.float32)
        right_6d = mat4_to_6d(self.R_mat) if self.R_mat is not None else np.zeros(6, dtype=np.float32)
        left_grip = getattr(self, 'left_gripper_state', GRIP_OPEN)
        right_grip = getattr(self, 'right_gripper_state', GRIP_OPEN)
        cmd = self.action_cmd
        return np.concatenate([
            left_6d, [left_grip], right_6d, [right_grip],
            [cmd["vx"], cmd["vy"], cmd["vyaw"], cmd["height"]],
        ]).astype(np.float32)

    def query_policy(self, img_mid, img_d435):
        """Send current observation to the policy server and return predicted action."""
        if self.policy is None:
            return None

        # Ego view = the d435 render at its native resolution (640x480, like
        # the training videos); the server resizes as in training.
        state = self._policy_state()
        obs = {
            "image": np.asarray(img_d435, dtype=np.uint8),
            "state": state,
            "prompt": getattr(self, 'policy_prompt', ""),
            "reset": getattr(self, '_policy_was_disabled', False),
        }
        if getattr(self, '_policy_was_disabled', False):
            print(f"[PolicyToggle]              state→server: {_fmt_action(state)}")
        result = self.policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)  # (horizon, 18)
        if actions.ndim != 2 or actions.shape[1] != 18:
            raise ValueError(f"Expected an (horizon, 18) action chunk, got {actions.shape}")
        return actions[:getattr(self, 'policy_chunk', len(actions))]

    def apply_policy_action(self, actions):
        """Apply the first action from the policy prediction to the robot."""
        action = actions[0]  # first step of the horizon
        self._last_applied_action = np.asarray(action, dtype=np.float32).copy()
        # action: [L_6d(6), L_grip(1), R_6d(6), R_grip(1), vx, vy, vyaw, height] = 18 dims

        # Apply left wrist (action[0:6]) as 4x4 matrix in robot-local frame
        left_6d = action[0:6]
        self.L_mat = np.eye(4, dtype=np.float32)
        self.L_mat[:3, 3] = left_6d[:3]
        self.L_mat[:3, :3] = R.from_euler('xyz', left_6d[3:6]).as_matrix()

        # Apply left gripper (action[6]): 0.5=open(lego), 1.0=closed(grab)
        self.left_gripper_state = float(np.clip(action[6], 0.0, 1.0))
        self.left_hand_vals = self._interpolate_hand(self.left_gripper_state, is_left_hand=True)

        # Apply right wrist (action[7:13]) as 4x4 matrix in robot-local frame
        right_6d = action[7:13]
        self.R_mat = np.eye(4, dtype=np.float32)
        self.R_mat[:3, 3] = right_6d[:3]
        self.R_mat[:3, :3] = R.from_euler('xyz', right_6d[3:6]).as_matrix()

        # Apply right gripper (action[13]): 0.5=open(lego), 1.0=closed(grab)
        self.right_gripper_state = float(np.clip(action[13], 0.0, 1.0))
        self.right_hand_vals = self._interpolate_hand(self.right_gripper_state, is_left_hand=False)

        # Sync viser wrist handles to policy output
        self.gui.left_wrist_handle.position = self.L_mat[:3, 3]
        self.gui.left_wrist_handle.wxyz = tft.quaternion_from_matrix(self.L_mat)
        self.gui.right_wrist_handle.position = self.R_mat[:3, 3]
        self.gui.right_wrist_handle.wxyz = tft.quaternion_from_matrix(self.R_mat)

        # Apply locomotion commands
        if len(action) >= 18:
            self.action_cmd = {
                "vx": float(action[14]),
                "vy": float(action[15]),
                "vyaw": float(action[16]),
                "height": float(action[17]),
            }

    def _update_action_chunk_plot(self, actions, current_step=0):
        """Update a live matplotlib window showing the current action chunk (loco dims)."""
        if not getattr(self.gui, 'show_chunk_plot', None) or not self.gui.show_chunk_plot.value:
            return

        import matplotlib.pyplot as plt

        if actions is None or len(actions) < 1 or actions.shape[-1] < 18:
            return

        if not hasattr(self, '_chunk_fig'):
            plt.ion()
            self._chunk_fig, self._chunk_axes = plt.subplots(3, 1, figsize=(4, 5), sharex=True)
            self._chunk_fig.canvas.manager.set_window_title("Action Chunk (loco)")

        axes = self._chunk_axes
        for ax in axes:
            ax.clear()

        horizon = len(actions)
        x = np.arange(horizon)
        loco = actions[:, 14:18]  # vx, vy, vyaw, height

        # Row 1: vx + vy
        axes[0].plot(x, loco[:, 0], '-o', color='#1f77b4', label='vx', markersize=3, linewidth=1.2)
        axes[0].plot(x, loco[:, 1], '-o', color='#ff7f0e', label='vy', markersize=3, linewidth=1.2)
        axes[0].set_ylim(-0.3, 0.3)
        axes[0].set_ylabel("m/s")
        axes[0].legend(loc="lower right", ncol=2, fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Row 2: vyaw
        axes[1].plot(x, loco[:, 2], '-o', color='#2ca02c', label='vyaw', markersize=3, linewidth=1.2)
        axes[1].set_ylim(-1.0, 1.0)
        axes[1].set_ylabel("rad/s")
        axes[1].legend(loc="lower right", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Row 3: height
        axes[2].plot(x, loco[:, 3], '-o', color='#d62728', label='height', markersize=3, linewidth=1.2)
        axes[2].set_ylim(0.6, 0.8)
        axes[2].set_ylabel("m")
        axes[2].set_xlabel("step")
        axes[2].legend(loc="lower right", fontsize=8)
        axes[2].grid(True, alpha=0.3)

        # Current step marker on all subplots
        if 0 <= current_step < horizon:
            for ax in axes:
                ax.axvline(current_step, color='red', linestyle='--', linewidth=1, alpha=0.7)

        self._chunk_fig.tight_layout()
        self._chunk_fig.canvas.draw_idle()
        self._chunk_fig.canvas.flush_events()

    def _interpolate_hand(self, grip_val, is_left_hand):
        """Interpolate hand joint values between stretch (0.0) and grab (1.0)."""
        base = np.array([0.0, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5])
        stretch = base if is_left_hand else -base
        grab = -base if is_left_hand else base
        t = np.clip(grip_val, 0.0, 1.0)
        return (stretch * (1.0 - t) + grab * t).tolist()

    def _run_policy_step(self):
        """Execute one policy inference/action step. Called from the main run loop."""
        if self.policy is not None and self.gui.policy_enabled:
            has_imgs = hasattr(self, '_last_img_mid') and hasattr(self, '_last_img_d435')
            if not has_imgs:
                self.gui.policy_status.content = "Policy: **waiting for rendered images...**"
            elif self.gui.policy_use_rtc:
                # --- RTC mode: async inference, temporal ensemble ---
                if not self._rtc_inflight:
                    self._rtc_inflight = True
                    img_mid = self._last_img_mid.copy()
                    img_d435 = self._last_img_d435.copy()
                    submit_tick = self._rtc_tick

                    def _infer():
                        try:
                            actions = self.query_policy(img_mid, img_d435)
                            if actions is not None:
                                with self._rtc_lock:
                                    self._rtc_chunk_list.append((submit_tick, actions))
                                if getattr(self, '_policy_was_disabled', False):
                                    last = getattr(self, '_last_applied_action', None)
                                    print(f"[PolicyToggle] ON (RTC) — new chunk[0]: {_fmt_action(actions[0])}")
                                    print(f"[PolicyToggle]              last applied: {_fmt_action(last)}")
                                    if last is not None and len(actions[0]) == len(last):
                                        diff = float(np.linalg.norm(np.asarray(actions[0]) - last))
                                        print(f"[PolicyToggle]              L2 diff: {diff:.4f}")
                                    self._policy_was_disabled = False
                        except Exception:
                            import traceback; traceback.print_exc()
                        finally:
                            self._rtc_inflight = False

                    threading.Thread(target=_infer, daemon=True).start()

                # Ensemble overlapping chunks
                t = self._rtc_tick
                k = self.gui.rtc_exp_weight.value
                contributions = []
                with self._rtc_lock:
                    alive = []
                    for (start, chunk) in self._rtc_chunk_list:
                        idx = t - start
                        if 0 <= idx < len(chunk):
                            alive.append((start, chunk))
                            contributions.append(chunk[idx])
                    self._rtc_chunk_list = alive

                if contributions:
                    n = len(contributions)
                    w = np.array([np.exp(-k * (n - 1 - j)) for j in range(n)])
                    w /= w.sum()
                    action = sum(wi * a for wi, a in zip(w, contributions))
                    action[6] = contributions[-1][6]    # L_grip: discrete, use latest
                    action[13] = contributions[-1][13]  # R_grip: discrete, use latest
                    self.apply_policy_action(action[np.newaxis])
                    loco_str = f"vx={action[14]:.2f} vy={action[15]:.2f} vyaw={action[16]:.2f} h={action[17]:.2f}" if len(action) >= 18 else ""
                    self.gui.policy_status.content = (
                        f"Policy: **RTC** tick={t} chunks={n} "
                        f"| {loco_str}"
                    )
                    # Show the latest chunk in the plot
                    if alive:
                        latest_chunk = alive[-1][1]
                        latest_idx = t - alive[-1][0]
                        self._update_action_chunk_plot(latest_chunk, current_step=latest_idx)

                self._rtc_tick += 1
            else:
                # --- Sequential mode: sync inference, execute full chunk ---
                if (self._rtc_chunks is None or
                        self._rtc_tick >= len(self._rtc_chunks)):
                    self.gui.policy_status.content = "Policy: **inferring...**"
                    try:
                        actions = self.query_policy(self._last_img_mid, self._last_img_d435)
                        if actions is not None:
                            self._rtc_chunks = actions
                            self._rtc_tick = 0
                            print(f"[Policy] New chunk: {len(actions)} steps")
                            if getattr(self, '_policy_was_disabled', False):
                                last = getattr(self, '_last_applied_action', None)
                                print(f"[PolicyToggle] ON (seq) — new chunk[0]: {_fmt_action(actions[0])}")
                                print(f"[PolicyToggle]              last applied: {_fmt_action(last)}")
                                if last is not None and len(actions[0]) == len(last):
                                    diff = float(np.linalg.norm(np.asarray(actions[0]) - last))
                                    print(f"[PolicyToggle]              L2 diff: {diff:.4f}")
                                self._policy_was_disabled = False
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        self.gui.policy_status.content = f"Policy: **error** {e}"

                if (self._rtc_chunks is not None and
                        self._rtc_tick < len(self._rtc_chunks)):
                    action = self._rtc_chunks[self._rtc_tick]
                    self.apply_policy_action(self._rtc_chunks[self._rtc_tick:])
                    self._update_action_chunk_plot(self._rtc_chunks, current_step=self._rtc_tick)
                    self._rtc_tick += 1
                    loco_str = f"vx={action[14]:.2f} vy={action[15]:.2f} vyaw={action[16]:.2f} h={action[17]:.2f}" if len(action) >= 18 else ""
                    self.gui.policy_status.content = (
                        f"Policy: **seq** [{self._rtc_tick}/{len(self._rtc_chunks)}] "
                        f"| {loco_str}"
                    )
        else:
            # Reset all state when policy is disabled
            if not getattr(self, '_policy_was_disabled', True):
                last = getattr(self, '_last_applied_action', None)
                print(f"[PolicyToggle] OFF — last applied: {_fmt_action(last)}")
                self._policy_was_disabled = True
            self._rtc_chunks = None
            self._rtc_tick = 0
            self._rtc_inflight = False
            with self._rtc_lock:
                self._rtc_chunk_list = []
