# Standard Python libraries
import threading

# Third-party libraries
import numpy as np
import rclpy
import torch
import time
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
    A ROS2 node that controls a robot arm and gripper using relative movements
    calculated by the OpenVLA model. This version uses a main loop for periodic control.
    """

    def __init__(self):
        super().__init__('vla_controller_node')

        # Declare and get parameters
        # self.declare_parameter('model_path', 'home/kist/openvla/ckpt/openvla-7b-base-bfloat16')
        self.declare_parameter('model_path', '/home/kist/openvla/ckpt/ecot-openvla-7b-bridge')
        self.declare_parameter('eef_pose_topic', '/ee_pose_cmd')
        self.declare_parameter('current_pose_topic', '/franka_robot_state_broadcaster/current_pose')
        self.declare_parameter('camera_topic', '/camera2/color/image_raw')
        self.declare_parameter('gripper_open_width', 0.08)
        self.declare_parameter('gripper_move_speed', 0.1)
        self.declare_parameter('gripper_grasp_force', 10.0)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        eef_pose_topic = self.get_parameter('eef_pose_topic').get_parameter_value().string_value
        current_pose_topic = self.get_parameter('current_pose_topic').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.gripper_open_width = self.get_parameter('gripper_open_width').get_parameter_value().double_value
        self.gripper_move_speed = self.get_parameter('gripper_move_speed').get_parameter_value().double_value
        self.gripper_grasp_force = self.get_parameter('gripper_grasp_force').get_parameter_value().double_value

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
        # self.prompt = "In: What action should the robot take to {pick blue block on the green plate and place it on the yellow plate}?\nOut:"

        self.instruction = "Pick a blue cube on the table and place into the pink cup"
        self.prompt = "A chat between a curious user and an artificial intelligence assistant. " + \
            "The assistant gives helpful, detailed, and polite answers to the user's questions. " + \
            f"USER: What action should the robot take to {self.instruction.lower()}? ASSISTANT: TASK:"

        self.get_logger().info(f"Model loaded successfully on device: {self.device}")
        # ---------------------------------------------------------

        # -------------------- ROS2 I/O SETUP --------------------
        self.get_logger().info("Setting up ROS2 publishers, subscribers, and action clients...")
        self.pose_pub = self.create_publisher(PoseStamped, eef_pose_topic, 10)
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, current_pose_topic, self.current_pose_callback, 10)
        self.move_cli = ActionClient(self, Move, "/fr3_gripper/move")
        self.grasp_cli = ActionClient(self, Grasp, "/fr3_gripper/grasp")
        self.get_logger().info("Waiting for gripper action servers...")
        self.move_cli.wait_for_server()
        self.grasp_cli.wait_for_server()
        self.get_logger().info("Gripper action servers are ready.")
        # --------------------------------------------------------

        # State and Utilities
        self.bridge = CvBridge()
        self.latest_image: PILImage.Image = None
        self.current_pose: PoseStamped = None
        self.image_lock = threading.Lock()
        self.pose_lock = threading.Lock()

        self.get_logger().info("VLA Controller node initialized. Control loop will run in main.")

    def image_callback(self, msg: Image):
        """Callback to receive and store the latest image from the camera."""
        with self.image_lock:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = PILImage.fromarray(cv_image)

    def current_pose_callback(self, msg: PoseStamped):
        """Callback to receive and store the latest robot EEF pose."""
        with self.pose_lock:
            self.current_pose = msg

    def control_loop(self):
        """The main logic for inference and relative action publishing."""
        # --- Step 1: Get Latest State from Shared Variables ---
        with self.image_lock:
            if self.latest_image is None:
                return # Wait for the first image
            local_image = self.latest_image

        with self.pose_lock:
            if self.current_pose is None:
                # This log might be spammy in a fast loop, consider removing or throttling
                # self.get_logger().info("Waiting for the first robot pose message...")
                return
            local_pose = self.current_pose
        # -------------------------------------------------

        # --- Step 2: VLA Model Inference ---
        inputs = self.processor(self.prompt, local_image, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            # CORRECTED: The model.predict_action function returns a numpy.ndarray directly.
            # No need for .cpu().numpy() conversion.
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        # -----------------------------------

        # --- Step 3: Calculate New Target Pose (Relative Control) ---
        eef_action_delta = action[:6]
        gripper_action_val = action[6]
        target_pose_msg = self.calculate_target_pose(local_pose, eef_action_delta)
        # ----------------------------------------------------------

        # --- Step 4: Publish Actions to Robot ---
        self.pose_pub.publish(target_pose_msg)
        print(f"Predicted Delta: {np.round(eef_action_delta, 3)}, Gripper: {gripper_action_val:.2f}")

        if gripper_action_val > 0.5:
            self.open_gripper()
        else:
            self.close_gripper()
        # ----------------------------------------

    def calculate_target_pose(self, current_pose: PoseStamped, delta: np.ndarray) -> PoseStamped:
        """
        Calculates the new target pose by adding a delta to the current pose.
        """
        # --- Position Calculation ---
        new_pos_x = current_pose.pose.position.x + delta[0]
        new_pos_y = current_pose.pose.position.y + delta[1]
        new_pos_z = current_pose.pose.position.z + delta[2]

        # --- Orientation Calculation ---
        current_quat = current_pose.pose.orientation
        current_rot = R.from_quat([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
        delta_rot = R.from_euler('xyz', delta[3:6], degrees=False)
        new_rot = current_rot * delta_rot
        new_quat = new_rot.as_quat()

        # --- Construct New PoseStamped Message ---
        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = current_pose.header.frame_id
        target_pose.pose.position.x = new_pos_x
        target_pose.pose.position.y = new_pos_y
        target_pose.pose.position.z = new_pos_z
        target_pose.pose.orientation.x = new_quat[0]
        target_pose.pose.orientation.y = new_quat[1]
        target_pose.pose.orientation.z = new_quat[2]
        target_pose.pose.orientation.w = new_quat[3]
        return target_pose

    def open_gripper(self):
        """Sends a goal to the 'move' action server to open the gripper."""
        goal_msg = Move.Goal()
        goal_msg.width = self.gripper_open_width
        goal_msg.speed = self.gripper_move_speed
        self.move_cli.send_goal_async(goal_msg)

    def close_gripper(self):
        """Sends a goal to the 'grasp' action server to close the gripper."""
        goal_msg = Grasp.Goal()
        goal_msg.force = self.gripper_grasp_force
        goal_msg.epsilon.inner = 0.005
        goal_msg.epsilon.outer = 0.005
        goal_msg.speed = self.gripper_move_speed
        self.grasp_cli.send_goal_async(goal_msg)


def main(args=None):
    rclpy.init(args=args)
    vla_controller_node = VLAControllerNode()

    try:
        # The main loop
        while rclpy.ok():
            start_time = time.time()
            # Process a single event (e.g., a callback)
            rclpy.spin_once(vla_controller_node, timeout_sec=0)

            # Execute the main control logic
            vla_controller_node.control_loop()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"duration: {elapsed_time}, Hz: {1/elapsed_time}")


            
    except KeyboardInterrupt:
        pass
    finally:
        vla_controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
