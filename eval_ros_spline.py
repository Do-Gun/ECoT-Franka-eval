# Standard Python libraries
import threading
import time

# Third-party libraries
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from franka_msgs.action import Move, Grasp
from geometry_msgs.msg import PoseStamped
from PIL import Image as PILImage
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class VLAControllerNode(Node):
    """
    Implements a velocity control scheme using a multi-threaded architecture.
    - An inference thread (~4Hz) determines a target action delta (a velocity command).
    - A control thread (1kHz) continuously applies a fraction of this delta to the
      robot's latest, real-time pose.
    - The main thread processes ROS callbacks to ensure the pose is always up-to-date.
    """

    def __init__(self):
        super().__init__('vla_controller_node')

        # Declare and get parameters
        self.declare_parameter('model_path', 'openvla-7b-v0.1')
        self.declare_parameter('eef_pose_topic', '/ee_pose_cmd')
        self.declare_parameter('current_pose_topic', '/franka_robot_state_broadcaster/current_pose')
        self.declare_parameter('camera_topic', '/camera2/color/image_raw')
        self.declare_parameter('gripper_open_width', 0.08)
        self.declare_parameter('gripper_move_speed', 0.1)
        self.declare_parameter('gripper_grasp_force', 10.0)
        self.declare_parameter('control_hz', 1000.0)
        self.declare_parameter('inference_hz', 4.0)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        eef_pose_topic = self.get_parameter('eef_pose_topic').get_parameter_value().string_value
        current_pose_topic = self.get_parameter('current_pose_topic').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.gripper_open_width = self.get_parameter('gripper_open_width').get_parameter_value().double_value
        self.gripper_move_speed = self.get_parameter('gripper_move_speed').get_parameter_value().double_value
        self.gripper_grasp_force = self.get_parameter('gripper_grasp_force').get_parameter_value().double_value
        self.control_hz = self.get_parameter('control_hz').get_parameter_value().double_value
        self.inference_hz = self.get_parameter('inference_hz').get_parameter_value().double_value

        # -------------------- INFERENCE SETUP --------------------
        self.get_logger().info("Setting up inference components...")
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        self.instruction = "Pick a pink cup on the table and place on the blue square plate"
        self.prompt = f"USER: What action should the robot take to {self.instruction.lower()}? ASSISTANT:"

        self.get_logger().info(f"Model loaded successfully on device: {self.device}")
        # ---------------------------------------------------------

        # -------------------- ROS2 I/O SETUP --------------------
        self.get_logger().info("Setting up ROS2 I/O...")
        self.pose_pub = self.create_publisher(PoseStamped, eef_pose_topic, 10)
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, current_pose_topic, self.current_pose_callback, 10)
        self.move_cli = ActionClient(self, Move, "/fr3_gripper/move")
        self.grasp_cli = ActionClient(self, Grasp, "/fr3_gripper/grasp")
        self.move_cli.wait_for_server()
        self.grasp_cli.wait_for_server()
        self.get_logger().info("ROS2 I/O is ready.")
        # --------------------------------------------------------

        # State and Utilities
        self.bridge = CvBridge()
        self.latest_image: PILImage.Image = None
        self.current_pose: PoseStamped = None
        
        # Shared variables for communication between threads
        self.latest_action_delta: np.ndarray = None
        self.latest_gripper_action: float = 0.0

        # Threading Locks
        self.image_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.action_lock = threading.Lock() # Lock for the action delta

        self.get_logger().info("VLA Controller node initialized.")

    def image_callback(self, msg: Image):
        with self.image_lock:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = PILImage.fromarray(cv_image)

    def current_pose_callback(self, msg: PoseStamped):
        with self.pose_lock:
            self.current_pose = msg

    def inference_worker(self):
        """
        [4Hz Loop] This worker runs VLA inference to produce a full action delta,
        which is then shared with the control worker.
        """
        self.get_logger().info("Inference worker started.")
        while rclpy.ok():
            with self.image_lock:
                local_image = self.latest_image
            
            if local_image is None:
                time.sleep(0.1)
                continue

            # Run VLA model inference
            inputs = self.processor(text=self.prompt, images=local_image, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
            with torch.no_grad():
                action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)


            # Control the gripper
            if action[6] > 0.5:
                self.open_gripper()
            else:
                self.close_gripper()

            # Safely update the shared action variables
            with self.action_lock:
                self.latest_action_delta = action[:6]
            
            self.get_logger().info(f"New action delta updated: {np.round(action[:6], 3)}")
            time.sleep(1.0 / self.inference_hz)

    def control_worker(self):
        """
        [1kHz Loop] This worker reads the latest action delta and applies a
        fraction of it to the robot's REAL-TIME current pose.
        """
        self.get_logger().info(f"Control worker started at {self.control_hz} Hz.")
        rate = self.create_rate(self.control_hz)

        while rclpy.ok():
            # Safely read the shared action delta
            with self.action_lock:
                local_delta = self.latest_action_delta

            # Safely read the real-time robot pose
            with self.pose_lock:
                realtime_pose = self.current_pose

            if local_delta is None or realtime_pose is None:
                rate.sleep()
                continue

            # Calculate the sub-delta for one 1kHz time step
            # This is the core logic: (action_delta / 250)
            sub_delta = local_delta * (self.inference_hz / self.control_hz)
            
            # Calculate the next target pose from the REAL-TIME pose
            target_pose_msg = self._calculate_pose_from_delta(realtime_pose, sub_delta)
            
            # Publish the action
            self.pose_pub.publish(target_pose_msg)
            
            rate.sleep()

    def _calculate_pose_from_delta(self, current_pose: PoseStamped, delta: np.ndarray) -> PoseStamped:
        # (Implementation is the same as before)
        new_pos_x = current_pose.pose.position.x + delta[0]
        new_pos_y = current_pose.pose.position.y + delta[1]
        new_pos_z = current_pose.pose.position.z + delta[2]
        current_rot = R.from_quat([current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w])
        delta_rot = R.from_euler('xyz', delta[3:6], degrees=False)
        new_rot = current_rot * delta_rot
        new_quat = new_rot.as_quat()
        target_pose = PoseStamped()
        target_pose.header = current_pose.header
        target_pose.pose.orientation.x, target_pose.pose.orientation.y, target_pose.orientation.z, target_pose.pose.orientation.w = new_quat
        target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z = new_pos_x, new_pos_y, new_pos_z
        return target_pose

    def open_gripper(self):
        # (Implementation is the same as before)
        goal_msg = Move.Goal()
        goal_msg.width = self.gripper_open_width
        goal_msg.speed = self.gripper_move_speed
        self.move_cli.send_goal_async(goal_msg)

    def close_gripper(self):
        # (Implementation is the same as before)
        goal_msg = Grasp.Goal()
        goal_msg.force = self.gripper_grasp_force
        goal_msg.epsilon.inner = 0.005
        goal_msg.epsilon.outer = 0.005
        goal_msg.speed = self.gripper_move_speed
        self.grasp_cli.send_goal_async(goal_msg)


def main(args=None):
    rclpy.init(args=args)
    vla_controller_node = VLAControllerNode()

    # Create and start the two worker threads
    inference_thread = threading.Thread(target=vla_controller_node.inference_worker, daemon=True)
    control_thread = threading.Thread(target=vla_controller_node.control_worker, daemon=True)
    inference_thread.start()
    control_thread.start()

    try:
        # The main thread spins to process ROS callbacks (image, pose) in real-time
        rclpy.spin(vla_controller_node)
    except KeyboardInterrupt:
        vla_controller_node.get_logger().info("Keyboard interrupt received, shutting down.")
    finally:
        vla_controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()