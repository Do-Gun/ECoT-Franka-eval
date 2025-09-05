# Standard Python libraries
import threading
import re
import json
import os
import enum

# Third-party libraries
import numpy as np
import rclpy
import torch
import time
from cv_bridge import CvBridge
from franka_msgs.action import Move, Grasp
from geometry_msgs.msg import PoseStamped
from PIL import Image as PILImage, ImageDraw
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from pathlib import Path


# === Official ECoT Reasoning Parsing (from Example.ipynb) ===
def split_reasoning(text, tags):
    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
            else:
                new_parts[k] = v

    return new_parts


class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]


def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        gripper_pos = [int(x) for x in gripper_pos.split(",")]
        gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
        metadata["gripper"] = gripper_pos

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            obj = sample.split("[")[0]
            if obj == "":
                continue
            coords = [int(n) for n in sample.split("[")[-1].split(",")]
            metadata["bboxes"][obj] = coords

    return metadata


class VLAControllerCoTNode(Node):
    """
    A ROS2 node that controls a robot arm and gripper using relative movements
    calculated by the OpenVLA model with Chain of Thought (CoT) reasoning visible.
    Combined with official ECoT implementation details.
    """

    def __init__(self):
        super().__init__('vla_controller_cot_node')

        # Declare and get parameters
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
        self.model_path = model_path
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Load Dataset Statistics from Disk (official ECoT style)
        if os.path.isdir(self.model_path):
            stats_path = Path(self.model_path) / "dataset_statistics.json"
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    self.model.norm_stats = json.load(f)
                    self.get_logger().info("Loaded dataset statistics from disk")

        self.instruction = "Pick a yellow cup on the table and place on the black base"
        
        # Official ECoT prompt format (with TASK: like Example.ipynb)
        SYSTEM_PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
                        "The assistant gives helpful, detailed, and polite answers to the user's questions.")
        self.prompt = f"{SYSTEM_PROMPT} USER: What action should the robot take to {self.instruction.lower()}? ASSISTANT: TASK:"

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

        self.get_logger().info("VLA Controller CoT node initialized. Control loop will run in main.")

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
        """The main logic for inference with CoT reasoning and relative action publishing."""
        # --- Step 1: Get Latest State from Shared Variables ---
        with self.image_lock:
            if self.latest_image is None:
                return # Wait for the first image
            local_image = self.latest_image

        with self.pose_lock:
            if self.current_pose is None:
                return
            local_pose = self.current_pose
        # -------------------------------------------------

        # --- Step 2: VLA Model Inference with CoT ---
        # Resize image to model's expected size (224x224) and convert to RGB (official style)
        model_image = local_image.resize((224, 224)).convert("RGB")
        inputs = self.processor(self.prompt, model_image, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            # Get action and generated text in one call (like official example)
            action, generated_ids = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, max_new_tokens=1024)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Parse reasoning using official ECoT method
            tags = [f" {tag}" for tag in get_cot_tags_list()]
            reasoning = split_reasoning(generated_text, tags)
            
            # Display structured reasoning (like Example.ipynb)
            print(f"\n" + "="*60)
            print(f"EMBODIED CHAIN-OF-THOUGHT REASONING")
            print(f"="*60)
            
            # Display main reasoning tags in order
            main_tags = [' TASK:', ' PLAN:', ' SUBTASK REASONING:', ' SUBTASK:', 
                        ' MOVE REASONING:', ' MOVE:', ' VISIBLE OBJECTS:', ' GRIPPER POSITION:']
            
            for tag in main_tags:
                if tag in reasoning:
                    print(f"\n{tag}")
                    print(f"{reasoning[tag]}")
            
            print(f"\n⚡ FINAL ACTION:")
            print(f"{action}")
            print(f"="*60)
            
            # Extract metadata and visualize (like Example.ipynb)
            metadata = get_metadata(reasoning)
            self.visualize_bounding_boxes_ecot(model_image, metadata)
        # -----------------------------------

        # --- Step 3: Calculate New Target Pose (Relative Control) ---
        if hasattr(action, '__len__') and len(action) >= 7:
            eef_action_delta = action[:6]
            gripper_action_val = action[6]
        else:
            print("Action format error, using default values")
            eef_action_delta = [0, 0, 0, 0, 0, 0]
            gripper_action_val = 0.5
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
        # Scale factor to amplify small deltas
        # position_scale = 1
        
        # --- Position Calculation ---
        new_pos_x = current_pose.pose.position.x + delta[0] #* position_scale
        new_pos_y = current_pose.pose.position.y + delta[1] #* position_scale
        new_pos_z = current_pose.pose.position.z + delta[2] #* position_scale

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

    def visualize_bounding_boxes_ecot(self, image: PILImage.Image, metadata: dict):
        """Visualize bounding boxes and gripper position using official ECoT metadata."""
        print(f"\n{'─'*40}")
        print(f"VISUAL GROUNDING")
        print(f"{'─'*40}")
        
        bboxes = {}
        for k, v in metadata["bboxes"].items():
            if k[0] == ",":
                k = k[1:]
            bboxes[k.lstrip().rstrip()] = v
        
        if bboxes:
            print(f"DETECTED OBJECTS:")
            for obj_name, bbox in bboxes.items():
                print(f"  • {obj_name}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        
        gripper_pos = metadata["gripper"][-1] if metadata["gripper"] else [0, 0]
        print(f"GRIPPER POSITION: [{gripper_pos[0]}, {gripper_pos[1]}]")
        
        # Create visualization
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Draw bounding boxes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan']
        for i, (obj_name, bbox) in enumerate(bboxes.items()):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            draw.text((x1, max(0, y1-12)), obj_name.strip(), fill=color)
        
        # Draw gripper position
        if gripper_pos != [0, 0]:
            draw.ellipse([gripper_pos[0]-5, gripper_pos[1]-5, 
                         gripper_pos[0]+5, gripper_pos[1]+5], fill='yellow')
            draw.ellipse([gripper_pos[0]-3, gripper_pos[1]-3, 
                         gripper_pos[0]+3, gripper_pos[1]+3], fill='red')
        
        # Save visualization
        vis_image.save("ecot_reasoning_visual.jpg")
        print(f"Visual saved to: ecot_reasoning_visual.jpg")
        print(f"{'─'*40}\n")


def main(args=None):
    rclpy.init(args=args)
    vla_controller_cot_node = VLAControllerCoTNode()

    try:
        # The main loop (clean output without duration)아
        while rclpy.ok():
            # Process a single event (e.g., a callback)
            rclpy.spin_once(vla_controller_cot_node, timeout_sec=0)

            # Execute the main control logic
            vla_controller_cot_node.control_loop()

    except KeyboardInterrupt:
        pass
    finally:
        vla_controller_cot_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()