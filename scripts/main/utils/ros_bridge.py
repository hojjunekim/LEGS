import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import threading
import numpy as np

import msgpack
from std_msgs.msg import ByteMultiArray
from std_srvs.srv import Trigger

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    # Note: HandState_ is for the Dex3-1 dexterous hands.
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, OdoState_, HandState_
except ImportError:
    print("[Warning] unitree_sdk2py not installed; the SONIC DDS bridge will fail.")

# --- Constants for Hand Joints ---
L_HAND_JOINTS = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint"
]

R_HAND_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint"
]


class RosBridge(Node):
    """Bridge to the SONIC whole-body controller over Unitree DDS.

    Subscribes to body/hand telemetry (rt/lowstate, rt/odostate, rt/dex3/*),
    publishes the 18-D command (base velocity + upper-body IK targets) on
    ControlPolicy/upper_body_pose, and relays per-object sim poses.
    """

    def __init__(self, network_interface="lo"):
        super().__init__('legs_ros_bridge')

        # Locks
        self.state_lock = threading.Lock()
        self.poses_lock = threading.Lock()

        # Data Storage
        self.latest_joint_state = {}
        # Per-entity poses keyed by body name (from /simulation/pose/{name}).
        self.latest_poses = {}
        # Track subscriptions so we don't register the same topic twice.
        self._pose_subscriptions = {}

        self.groot_pos = np.array([0.0, 0.0, 0.0])
        self.groot_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

        # Strictly 29 body joints to avoid reading padding in lowstate.
        self.body_joint_names = [
            # --- Legs (0-11) ---
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",

            # --- Waist (12-14) ---
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",

            # --- Left Arm (15-21) ---
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",

            # --- Right Arm (22-28) ---
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]

        # Initialize native Unitree DDS on the specified interface.
        ChannelFactoryInitialize(0, network_interface)

        # --- Body Subscriptions ---
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.dds_lowstate_callback)

        self.odo_sub = ChannelSubscriber("rt/odostate", OdoState_)
        self.odo_sub.Init(self.dds_odostate_callback)

        # --- Hand Subscriptions ---
        self.left_hand_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        self.left_hand_sub.Init(self.dds_lefthand_callback)

        self.right_hand_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
        self.right_hand_sub.Init(self.dds_righthand_callback)

        self.groot_cmd_pub = self.create_publisher(ByteMultiArray, 'ControlPolicy/upper_body_pose', 1)
        print(f"[ROS Bridge] SONIC DDS bridge listening on {network_interface}")

        self.reset_client = self.create_client(Trigger, '/simulation/reset')

    def subscribe_pose(self, name: str):
        """Subscribe to /simulation/pose/{name}. Safe to call multiple times."""
        if name in self._pose_subscriptions:
            return
        topic = f"/simulation/pose/{name}"
        sub = self.create_subscription(
            PoseStamped, topic, lambda msg, n=name: self._pose_callback(n, msg), 10
        )
        self._pose_subscriptions[name] = sub
        print(f"[ROS Bridge] Subscribed to {topic}")

    def _pose_callback(self, name: str, msg):
        with self.poses_lock:
            self.latest_poses[name] = msg.pose

    def publish_full_command(self, vx, vy, vyaw, target_height,
                             L_mat=None, R_mat=None, head_mat=None,
                             left_hand=None, right_hand=None, control_toggle=False):
        """Publish base velocity + full upper-body IK targets to the SONIC policy."""
        locomotion_mode = 0 if target_height > 0.5 else 6

        command_dict = {
            "navigate_cmd": [float(vx), float(vy), float(vyaw)],
            "base_height_command": float(target_height),
            "locomotion_mode": locomotion_mode,
            "toggle_policy_action": control_toggle
        }

        # Append optional upper-body data if provided by the visualizer.
        if L_mat is not None:
            command_dict["left_wrist_after_ik"] = L_mat.tolist()
        if R_mat is not None:
            command_dict["right_wrist_after_ik"] = R_mat.tolist()
        if head_mat is not None:
            command_dict["head_after_ik"] = head_mat.tolist()
        if left_hand is not None:
            command_dict["left_hand_joint"] = left_hand
        if right_hand is not None:
            command_dict["right_hand_joint"] = right_hand

        packed_data = msgpack.packb(command_dict, use_bin_type=True)
        msg = ByteMultiArray()
        msg.data = [bytes([b]) for b in packed_data]

        self.groot_cmd_pub.publish(msg)

    # --- Unitree SDK DDS Callbacks ---
    def dds_lowstate_callback(self, msg: LowState_):
        """Body telemetry: IMU quaternion + 29 motor joints."""
        with self.state_lock:
            self.groot_quat = np.array([
                msg.imu_state.quaternion[0],
                msg.imu_state.quaternion[1],
                msg.imu_state.quaternion[2],
                msg.imu_state.quaternion[3]
            ])

            # Strictly limit to 29 joints to avoid reading zero-padding.
            for i in range(29):
                if i < len(msg.motor_state):
                    self.latest_joint_state[self.body_joint_names[i]] = msg.motor_state[i].q

    def dds_odostate_callback(self, msg: OdoState_):
        """Odometry: base position."""
        with self.state_lock:
            self.groot_pos = np.array([
                msg.position[0],
                msg.position[1],
                msg.position[2]
            ])

    def dds_lefthand_callback(self, msg: HandState_):
        """Left-hand (Dex3) joint telemetry."""
        with self.state_lock:
            for i, joint_name in enumerate(L_HAND_JOINTS):
                if i < len(msg.motor_state):
                    self.latest_joint_state[joint_name] = msg.motor_state[i].q

    def dds_righthand_callback(self, msg: HandState_):
        """Right-hand (Dex3) joint telemetry."""
        with self.state_lock:
            for i, joint_name in enumerate(R_HAND_JOINTS):
                if i < len(msg.motor_state):
                    self.latest_joint_state[joint_name] = msg.motor_state[i].q

    # --- Data Retrieval ---
    def get_latest_state(self):
        """Returns (position, quaternion_wxyz, joint_dict) from the SONIC DDS telemetry."""
        with self.state_lock:
            if not self.latest_joint_state:
                return None
            # Compiled dict contains body + both hands.
            return self.groot_pos.copy(), self.groot_quat.copy(), self.latest_joint_state.copy()

    def get_pose(self, name: str):
        """Return (pos_xyz, quat_wxyz) for the given entity, or None if not received yet."""
        with self.poses_lock:
            pose = self.latest_poses.get(name)
            if pose is None:
                return None
            p = pose.position
            q = pose.orientation
            pos = np.array([p.x, p.y, p.z])
            quat_wxyz = np.array([q.w, q.x, q.y, q.z])
            return pos, quat_wxyz

    def reset_simulation(self):
        """Call /simulation/reset service to reset MuJoCo sim."""
        if not self.reset_client.service_is_ready():
            print("[ROS Bridge] /simulation/reset service not available")
            return False
        self.reset_client.call_async(Trigger.Request())
        print("[ROS Bridge] Reset request sent to MuJoCo sim")
        return True


def start_ros_thread(network_interface="lo"):
    rclpy.init()
    node = RosBridge(network_interface=network_interface)
    # Spin the node on a separate thread so it doesn't block the main viz loop.
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return node
