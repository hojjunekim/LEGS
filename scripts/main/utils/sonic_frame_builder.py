"""
Build a per-frame dict matching the GR00T SONIC LeRobot schema
(`get_features_sonic_vla` + `get_modality_config_sonic_vla`) plus extra
Psi0 columns for the 18-D command the LEGS sim sends over ROS 2.

Source data:
    - per-frame record from ``DataRecorder`` (one entry of episode_XXXX.json)
    - scenario from ``episode_XXXX_scenario.json``  (joint_names, init_base_quat, ...)
    - rendered RGB image (uint8 [H=480, W=640, 3])
    - optional 64-d motion_token vector (filled offline from deploy CSV)

Joint ordering follows SONIC ``RobotModel.joint_names`` for G1 + ThreeFinger
(43 joints, groups: left_leg, right_leg, waist, left_arm, left_hand,
right_arm, right_hand). The LEGS sim records 29 body + 14 hand joints whose
NAMES are identical but order differs (it lists right_arm before the hands).
We resolve everything by joint name lookup, so order at the source doesn't
matter as long as every SONIC joint name is present in joints_dict.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


# Canonical SONIC G1 joint order (43 joints). Pulled from
# gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info.
SONIC_JOINT_ORDER: list[str] = [
    # left_leg (0-5)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # right_leg (6-11)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # waist (12-14)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # left_arm (15-21)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # left_hand (22-28) — thumb-first, matches MJCF / pinocchio /
    # g1_supplemental_info.left_hand_actuated_joints. Do NOT use the alphabetical
    # joint_groups["left_hand"]["joints"] order — that's a semantic grouping for
    # the masking API and silently permutes the SONIC encoder's gripper slots.
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    # right_arm (29-35)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # right_hand (36-42) — thumb-first (see note above)
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
NUM_JOINTS = len(SONIC_JOINT_ORDER)  # 43

# SONIC stream mode: 5 = VR_3pt + Planner. The LEGS sim collects demos by
# driving SONIC's ROS2 input handler with planner + wrist 6D commands, which
# is the deploy-time analog of stream_mode=5.
STREAM_MODE_VR3PT_PLANNER: int = 5

EGO_VIEW_HEIGHT: int = 480
EGO_VIEW_WIDTH: int = 640


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _wxyz_to_R(wxyz) -> np.ndarray:
    qw, qx, qy, qz = wxyz
    return R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)


def _R_to_wxyz(mat: np.ndarray) -> np.ndarray:
    q_xyzw = R.from_matrix(mat).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def _mat3_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """SONIC rotation_6d convention: [r00, r10, r01, r11, r02, r12]
    (top-two rows of the 3x3 rotation matrix). See feature ``names`` field
    in features_sonic_vla.target_body_orientation."""
    return np.array([
        mat[0, 0], mat[1, 0],
        mat[0, 1], mat[1, 1],
        mat[0, 2], mat[1, 2],
    ], dtype=np.float32)


def _eef_pos_quat_wxyz(T4: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if T4 is None:
        return np.zeros(3, dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32)
    pos = np.asarray(T4[:3, 3], dtype=np.float32)
    quat_wxyz = _R_to_wxyz(T4[:3, :3])
    return pos, quat_wxyz


def _eef_pos_eulerXYZ(T4: Optional[np.ndarray]) -> np.ndarray:
    """Psi0 6-D EEF representation: pos(3) + euler_xyz(3). Matches
    old/build_lerobot_dataset_psi_18act_eef.py::mat4_to_6d."""
    if T4 is None:
        return np.zeros(6, dtype=np.float32)
    pos = np.asarray(T4[:3, 3], dtype=np.float32)
    rpy = R.from_matrix(T4[:3, :3]).as_euler("xyz").astype(np.float32)
    return np.concatenate([pos, rpy])


def _wrist_rpy(T4: Optional[np.ndarray]) -> np.ndarray:
    """SONIC ``teleop.left/right_wrist_joints``: [roll, pitch, yaw] euler XYZ."""
    if T4 is None:
        return np.zeros(3, dtype=np.float32)
    return R.from_matrix(T4[:3, :3]).as_euler("xyz").astype(np.float32)


def _projected_gravity(base_wxyz) -> np.ndarray:
    """Unit gravity expressed in the body frame. World gravity points -Z."""
    R_body_to_world = _wxyz_to_R(base_wxyz)
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (R_body_to_world.T @ g_world).astype(np.float32)


def _facing_vector(base_wxyz) -> np.ndarray:
    """Body +X axis in world frame — used for SONIC ``teleop.planner_facing``."""
    return _wxyz_to_R(base_wxyz)[:, 0].astype(np.float32)


# ---------------------------------------------------------------------------
# SONIC joint vector
# ---------------------------------------------------------------------------


def joints_dict_to_sonic_vector(joints_dict: dict) -> np.ndarray:
    """Pack {joint_name: angle} into a 43-dim vector in SONIC's canonical
    order. Missing joints (e.g. if the recorder drops a finger) become 0.0
    rather than failing — the encoder is robust to small zero patches but
    relies on the slot ordering being correct.
    """
    out = np.zeros(NUM_JOINTS, dtype=np.float64)
    for i, name in enumerate(SONIC_JOINT_ORDER):
        v = joints_dict.get(name)
        if v is not None:
            out[i] = float(v)
    return out


# ---------------------------------------------------------------------------
# Frame builder
# ---------------------------------------------------------------------------


def build_sonic_frame(
    record: dict,
    scenario: dict,
    image_rgb_uint8: np.ndarray,
    motion_token: Optional[np.ndarray] = None,
    task: Optional[str] = None,
) -> dict:
    """Produce a frame dict ready for ``Gr00tDataExporter.add_frame``.

    All channels in ``get_features_sonic_vla`` + the Psi0 sidecar columns
    (``observation.state_psi0``, ``action.psi0_18``) are populated. Frames
    that lack the corresponding raw inputs (e.g. zero-fill for SMPL streams,
    motion_token before offline labeling) get zero arrays of the right
    dtype/shape so the parquet stays schema-stable across sim and teleop.

    :param record: one entry of ``episode_XXXX.json``.
    :param scenario: parsed ``episode_XXXX_scenario.json`` (uses
        ``joint_names`` and ``start_wxyz_w``).
    :param image_rgb_uint8: rendered ego_view (480, 640, 3) uint8.
    :param motion_token: 64-dim vector from the deploy's
        ``token_state.csv`` (None ⇒ zeros).
    :param task: language instruction (falls back to scenario prompt).
    """

    # --- raw inputs ---------------------------------------------------
    rob = record["robot"]
    base_pos = np.asarray(rob["base_pos_w"], dtype=np.float64)
    base_wxyz = np.asarray(rob["base_wxyz_w"], dtype=np.float64)

    src_joint_names = scenario.get("joint_names") or []
    joints_dict = {n: v for n, v in zip(src_joint_names, rob["joints"])}

    L_mat = np.asarray(record["L_mat"], dtype=np.float32) if record.get("L_mat") is not None else None
    R_mat = np.asarray(record["R_mat"], dtype=np.float32) if record.get("R_mat") is not None else None
    H_mat = np.asarray(record["H_mat"], dtype=np.float32) if record.get("H_mat") is not None else None

    cmd = record.get("action_cmd", {})
    vx = float(cmd.get("vx", 0.0))
    vy = float(cmd.get("vy", 0.0))
    vyaw = float(cmd.get("vyaw", 0.0))
    height = float(cmd.get("height", 0.78))
    speed = float(np.hypot(vx, vy))

    left_grip = float(record.get("left_gripper", 0.0))
    right_grip = float(record.get("right_gripper", 0.0))

    init_base_wxyz = np.asarray(scenario.get("start_wxyz_w", [1, 0, 0, 0]), dtype=np.float64)
    task_str = task or scenario.get("language_instruction") or ""

    # --- SONIC observation channels -----------------------------------
    state_43 = joints_dict_to_sonic_vector(joints_dict)

    L_pos, L_quat = _eef_pos_quat_wxyz(L_mat)
    R_pos, R_quat = _eef_pos_quat_wxyz(R_mat)
    eef_state_14 = np.concatenate([L_pos, L_quat, R_pos, R_quat]).astype(np.float64)

    # --- SONIC teleop channels ----------------------------------------
    body_R = _wxyz_to_R(base_wxyz)
    target_body_6d = _mat3_to_rot6d(body_R)

    L_pos3 = L_mat[:3, 3] if L_mat is not None else np.zeros(3, dtype=np.float32)
    R_pos3 = R_mat[:3, 3] if R_mat is not None else np.zeros(3, dtype=np.float32)
    H_pos3 = H_mat[:3, 3] if H_mat is not None else np.zeros(3, dtype=np.float32)
    vr_3pt_pos = np.concatenate([L_pos3, R_pos3, H_pos3]).astype(np.float32)

    L_R = L_mat[:3, :3] if L_mat is not None else np.eye(3, dtype=np.float32)
    R_R = R_mat[:3, :3] if R_mat is not None else np.eye(3, dtype=np.float32)
    H_R = H_mat[:3, :3] if H_mat is not None else np.eye(3, dtype=np.float32)
    vr_3pt_orient = np.concatenate([
        _mat3_to_rot6d(L_R), _mat3_to_rot6d(R_R), _mat3_to_rot6d(H_R)
    ]).astype(np.float32)

    # --- Psi0 sidecar columns -----------------------------------------
    L_eef_6d = _eef_pos_eulerXYZ(L_mat)
    R_eef_6d = _eef_pos_eulerXYZ(R_mat)
    state_psi0_15 = np.concatenate([
        L_eef_6d, [left_grip], R_eef_6d, [right_grip], [height]
    ]).astype(np.float32)
    action_psi0_18 = np.concatenate([
        L_eef_6d, [left_grip], R_eef_6d, [right_grip], [vx, vy, vyaw, height]
    ]).astype(np.float32)

    # Frame dict — every key MUST appear in ``get_features_sonic_vla`` (or
    # the Psi0 sidecar feature dict added in build_sonic_lerobot.py).
    frame: dict = {
        "task": task_str,

        # video
        "observation.images.ego_view": image_rgb_uint8,

        # SONIC observation
        "observation.state": state_43,
        "observation.eef_state": eef_state_14,
        "observation.root_orientation": np.asarray(base_wxyz, dtype=np.float64),
        "observation.projected_gravity": _projected_gravity(base_wxyz).astype(np.float64),
        "observation.cpp_rotation_offset": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "observation.init_base_quat": np.asarray(init_base_wxyz, dtype=np.float64),

        # SONIC actions
        "action.wbc": state_43.copy(),  # joint targets ≈ current state for sim imitation
        "action.motion_token": (
            np.asarray(motion_token, dtype=np.float64)
            if motion_token is not None
            else np.zeros(64, dtype=np.float64)
        ),

        # Teleop / planner stream (recorded so encoder fine-tune has the inputs)
        "teleop.delta_heading": np.array([0.0], dtype=np.float64),
        "teleop.smpl_joints": np.zeros(72, dtype=np.float32),
        "teleop.smpl_pose": np.zeros(63, dtype=np.float32),
        "teleop.body_quat_w": np.asarray(base_wxyz, dtype=np.float32),
        "teleop.target_body_orientation": target_body_6d,
        "teleop.left_hand_joints": state_43[22:29].astype(np.float32),
        "teleop.right_hand_joints": state_43[36:43].astype(np.float32),
        "teleop.smpl_frame_index": np.array([0], dtype=np.int64),
        "teleop.left_wrist_joints": _wrist_rpy(L_mat),
        "teleop.right_wrist_joints": _wrist_rpy(R_mat),
        "teleop.stream_mode": np.array([STREAM_MODE_VR3PT_PLANNER], dtype=np.int32),
        "teleop.planner_mode": np.array([1 if speed > 1e-3 or abs(vyaw) > 1e-3 else 0], dtype=np.int32),
        "teleop.planner_movement": np.array([vx, vy, vyaw], dtype=np.float32),
        "teleop.planner_facing": _facing_vector(base_wxyz),
        "teleop.planner_speed": np.array([speed], dtype=np.float32),
        "teleop.planner_height": np.array([height], dtype=np.float32),
        "teleop.vr_3pt_position": vr_3pt_pos,
        "teleop.vr_3pt_orientation": vr_3pt_orient,

        # Psi0 sidecar columns (training reads via modality_psi0.json)
        "observation.state_psi0": state_psi0_15,
        "action.psi0_18": action_psi0_18,
    }
    return frame


# ---------------------------------------------------------------------------
# Feature & modality dicts (universal — covers SONIC + Psi0)
# ---------------------------------------------------------------------------


def get_universal_features() -> dict:
    """SONIC features + Psi0 sidecar columns. Direct extension of
    ``gear_sonic.data.features_sonic_vla.get_features_sonic_vla`` — names and
    shapes match exactly so a Psi0-only consumer can ignore the SONIC
    columns and vice versa."""
    return {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float64",
            "shape": (NUM_JOINTS,),
            "names": SONIC_JOINT_ORDER,
        },
        "observation.eef_state": {
            "dtype": "float64",
            "shape": (14,),
            "names": ["left_wrist_pos", "left_wrist_abs_quat", "right_wrist_pos", "right_wrist_abs_quat"],
        },
        "action.wbc": {
            "dtype": "float64",
            "shape": (NUM_JOINTS,),
            "names": SONIC_JOINT_ORDER,
        },
        "observation.root_orientation": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["base_qw", "base_qx", "base_qy", "base_qz"],
        },
        "observation.projected_gravity": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["gravity_x", "gravity_y", "gravity_z"],
        },
        "observation.cpp_rotation_offset": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["rot_offset_qw", "rot_offset_qx", "rot_offset_qy", "rot_offset_qz"],
        },
        "observation.init_base_quat": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["init_base_qw", "init_base_qx", "init_base_qy", "init_base_qz"],
        },
        "teleop.delta_heading": {"dtype": "float64", "shape": (1,), "names": ["delta_heading"]},
        "action.motion_token": {"dtype": "float64", "shape": (64,), "names": "motion_token"},
        "teleop.smpl_joints": {"dtype": "float32", "shape": (72,), "names": "smpl_joints"},
        "teleop.smpl_pose": {"dtype": "float32", "shape": (63,), "names": "smpl_pose"},
        "teleop.body_quat_w": {"dtype": "float32", "shape": (4,), "names": "body_quat_w"},
        "teleop.target_body_orientation": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "target_body_r00", "target_body_r10",
                "target_body_r01", "target_body_r11",
                "target_body_r02", "target_body_r12",
            ],
        },
        "teleop.left_hand_joints": {"dtype": "float32", "shape": (7,), "names": "left_hand_joints"},
        "teleop.right_hand_joints": {"dtype": "float32", "shape": (7,), "names": "right_hand_joints"},
        "teleop.smpl_frame_index": {"dtype": "int64", "shape": (1,), "names": ["smpl_frame_index"]},
        "teleop.left_wrist_joints": {
            "dtype": "float32", "shape": (3,),
            "names": ["left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"],
        },
        "teleop.right_wrist_joints": {
            "dtype": "float32", "shape": (3,),
            "names": ["right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"],
        },
        "teleop.stream_mode": {"dtype": "int32", "shape": (1,), "names": ["stream_mode"]},
        "teleop.planner_mode": {"dtype": "int32", "shape": (1,), "names": ["locomotion_mode"]},
        "teleop.planner_movement": {
            "dtype": "float32", "shape": (3,),
            "names": ["movement_x", "movement_y", "movement_z"],
        },
        "teleop.planner_facing": {
            "dtype": "float32", "shape": (3,),
            "names": ["facing_x", "facing_y", "facing_z"],
        },
        "teleop.planner_speed": {"dtype": "float32", "shape": (1,), "names": ["speed"]},
        "teleop.planner_height": {"dtype": "float32", "shape": (1,), "names": ["height"]},
        "teleop.vr_3pt_position": {
            "dtype": "float32", "shape": (9,),
            "names": [
                "lwrist_x", "lwrist_y", "lwrist_z",
                "rwrist_x", "rwrist_y", "rwrist_z",
                "neck_x", "neck_y", "neck_z",
            ],
        },
        "teleop.vr_3pt_orientation": {
            "dtype": "float32", "shape": (18,),
            "names": [
                "lwrist_r00", "lwrist_r10", "lwrist_r01", "lwrist_r11", "lwrist_r02", "lwrist_r12",
                "rwrist_r00", "rwrist_r10", "rwrist_r01", "rwrist_r11", "rwrist_r02", "rwrist_r12",
                "neck_r00", "neck_r10", "neck_r01", "neck_r11", "neck_r02", "neck_r12",
            ],
        },

        # Psi0 sidecar columns (15-dim state, 18-dim action; see
        # old/build_lerobot_dataset_psi_18act_eef.py for the canonical layout).
        "observation.state_psi0": {
            "dtype": "float32", "shape": (15,),
            "names": [
                "L_eef_x", "L_eef_y", "L_eef_z", "L_eef_roll", "L_eef_pitch", "L_eef_yaw",
                "L_grip",
                "R_eef_x", "R_eef_y", "R_eef_z", "R_eef_roll", "R_eef_pitch", "R_eef_yaw",
                "R_grip",
                "height",
            ],
        },
        "action.psi0_18": {
            "dtype": "float32", "shape": (18,),
            "names": [
                "L_eef_x", "L_eef_y", "L_eef_z", "L_eef_roll", "L_eef_pitch", "L_eef_yaw",
                "L_grip",
                "R_eef_x", "R_eef_y", "R_eef_z", "R_eef_roll", "R_eef_pitch", "R_eef_yaw",
                "R_grip",
                "vx", "vy", "vyaw", "height",
            ],
        },
    }


def get_modality_config_sonic() -> dict:
    """SONIC training modality (matches gear_sonic.data.features_sonic_vla)."""
    return {
        "state": {
            "left_leg":  {"start": 0,  "end": 6},
            "right_leg": {"start": 6,  "end": 12},
            "waist":     {"start": 12, "end": 15},
            "left_arm":  {"start": 15, "end": 22},
            "left_hand": {"start": 22, "end": 29},
            "right_arm": {"start": 29, "end": 36},
            "right_hand":{"start": 36, "end": 43},
            "left_wrist_pos":      {"start": 0, "end": 3,  "original_key": "observation.eef_state"},
            "left_wrist_abs_quat": {"start": 3, "end": 7,  "original_key": "observation.eef_state", "rotation_type": "quaternion"},
            "right_wrist_pos":     {"start": 7, "end": 10, "original_key": "observation.eef_state"},
            "right_wrist_abs_quat":{"start": 10,"end": 14, "original_key": "observation.eef_state", "rotation_type": "quaternion"},
            "root_orientation":    {"start": 0, "end": 4,  "original_key": "observation.root_orientation", "rotation_type": "quaternion"},
            "projected_gravity":   {"start": 0, "end": 3,  "original_key": "observation.projected_gravity"},
            "cpp_rotation_offset": {"start": 0, "end": 4,  "original_key": "observation.cpp_rotation_offset", "rotation_type": "quaternion"},
            "init_base_quat":      {"start": 0, "end": 4,  "original_key": "observation.init_base_quat", "rotation_type": "quaternion"},
        },
        "action": {
            "delta_heading":           {"start": 0, "end": 1,  "original_key": "teleop.delta_heading"},
            "motion_token":            {"start": 0, "end": 64, "original_key": "action.motion_token"},
            "smpl_joints":             {"start": 0, "end": 72, "original_key": "teleop.smpl_joints"},
            "smpl_pose":               {"start": 0, "end": 63, "original_key": "teleop.smpl_pose"},
            "body_quat_w":             {"start": 0, "end": 4,  "original_key": "teleop.body_quat_w", "rotation_type": "quaternion"},
            "target_body_orientation": {"start": 0, "end": 6,  "original_key": "teleop.target_body_orientation", "rotation_type": "rotation_6d"},
            "left_hand_joints":        {"start": 0, "end": 7,  "original_key": "teleop.left_hand_joints"},
            "right_hand_joints":       {"start": 0, "end": 7,  "original_key": "teleop.right_hand_joints"},
            "left_wrist_joints":       {"start": 0, "end": 3,  "original_key": "teleop.left_wrist_joints"},
            "right_wrist_joints":      {"start": 0, "end": 3,  "original_key": "teleop.right_wrist_joints"},
            "stream_mode":             {"start": 0, "end": 1,  "original_key": "teleop.stream_mode"},
            "planner_mode":            {"start": 0, "end": 1,  "original_key": "teleop.planner_mode"},
            "planner_movement":        {"start": 0, "end": 3,  "original_key": "teleop.planner_movement"},
            "planner_facing":          {"start": 0, "end": 3,  "original_key": "teleop.planner_facing"},
            "planner_speed":           {"start": 0, "end": 1,  "original_key": "teleop.planner_speed"},
            "planner_height":          {"start": 0, "end": 1,  "original_key": "teleop.planner_height"},
            "vr_3pt_position":         {"start": 0, "end": 9,  "original_key": "teleop.vr_3pt_position"},
            "vr_3pt_orientation":      {"start": 0, "end": 18, "original_key": "teleop.vr_3pt_orientation", "rotation_type": "rotation_6d"},
        },
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def get_modality_config_psi0() -> dict:
    """Psi0 18-dim ROS2 head modality. Reads only the Psi0 sidecar columns
    plus the egocentric view + task. Informational: the Psi0 trainer selects
    its columns through command-line keys (scripts/train/psi0.sh), not this
    file."""
    return {
        "state": {"all": {"start": 0, "end": 15, "original_key": "observation.state_psi0"}},
        "action": {"all": {"start": 0, "end": 18, "original_key": "action.psi0_18"}},
        "video": {"egocentric": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
