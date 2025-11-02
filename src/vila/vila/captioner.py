import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time

from vila_msgs.msg import CaptionWithPose

import os
import cv2
import math
import time
import threading
import traceback
import statistics
import concurrent.futures

from PIL import Image as im
from transformers import GenerationConfig

import llava
from llava import conversation as clib
from llava.media import Image
from llava.utils.logging import logger


def configure_ps3_and_context_length(model):
    """Configure PS3 settings and adjust context length based on environment variables."""
    # Get PS3 configs from environment variables
    num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
    num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
    select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
    look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
    smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

    # Set PS3 configs
    if num_look_close is not None:
        logger.info(f"Num look close: {num_look_close}")
        model.num_look_close = int(num_look_close)
    if num_token_look_close is not None:
        logger.info(f"Num token look close: {num_token_look_close}")
        model.num_token_look_close = int(num_token_look_close)
    if select_num_each_scale is not None:
        logger.info(f"Select num each scale: {select_num_each_scale}")
        select_num_each_scale = [int(x)
                                 for x in select_num_each_scale.split("+")]
        model.get_vision_tower(
        ).vision_tower.vision_model.max_select_num_each_scale = select_num_each_scale
    if look_close_mode is not None:
        logger.info(f"Look close mode: {look_close_mode}")
        model.look_close_mode = look_close_mode
    if smooth_selection_prob is not None:
        logger.info(f"Smooth selection prob: {smooth_selection_prob}")
        if smooth_selection_prob.lower() == "true":
            smooth_selection_prob = True
        elif smooth_selection_prob.lower() == "false":
            smooth_selection_prob = False
        else:
            raise ValueError(
                f"Invalid smooth selection prob: {smooth_selection_prob}")
        model.smooth_selection_prob = smooth_selection_prob

    # Adjust the max context length based on the PS3 config
    context_length = model.tokenizer.model_max_length
    if num_look_close is not None:
        context_length = max(context_length, int(
            num_look_close) * 2560 // 4 + 1024)
    if num_token_look_close is not None:
        context_length = max(context_length, int(
            num_token_look_close) // 4 + 1024)
    context_length = max(
        getattr(model.tokenizer, "model_max_length", context_length), context_length)
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length


class Captioner(Node):
    # Maximum buffer size to prevent unbounded memory growth
    MAX_BUFFER_SIZE = 100

    def __init__(self,
                 model_path="NVILA-Lite-8B",
                 prompt_text="Describe what you see in these images.",
                 conv_mode="vicuna_v1"):
        super().__init__('captioner')

        # Image subscriber for compressed image data
        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.image_callback,
            10)
        self.image_subscriber  # prevent unused variable warning

        # Image buffer stores (image, timestamp) tuples
        self.image_buffer = []

        # Odometry subscriber for robot pose data
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.odom_subscriber  # prevent unused variable warning

        # Odometry buffer stores (timestamp, x, y, z, theta) tuples
        self.odom_buffer = []

        # Track last known pose as fallback when odom buffer is empty
        self.last_pose = None

        # Publisher for caption with pose data
        self.caption_pose_pub = self.create_publisher(
            CaptionWithPose,
            '/caption_with_pose',
            10)
        logger.info("CaptionWithPose publisher created on /caption_with_pose")

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Thread safety lock for shared state
        self.buffer_lock = threading.Lock()

        # Initialize to None to handle first message properly
        self.start_time = None
        self.segment_time = 3  # seconds

        self.vila_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        # Store prompt text for VILA queries
        self.prompt_text = prompt_text

        # Load VILA model
        logger.info(f"Loading VILA model from: {model_path}")
        self.model = llava.load(model_path, model_base=None)
        logger.info("VILA model loaded successfully")

        # Configure PS3 and context length
        configure_ps3_and_context_length(self.model)

        # Set conversation mode
        clib.default_conversation = clib.conv_templates[conv_mode].copy()
        logger.info(f"Using conversation mode: {conv_mode}")

    def image_callback(self, msg):
        """Process incoming image messages with proper error handling and thread safety."""
        try:
            # Convert ROS CompressedImage to OpenCV image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)

            # Convert BGR to RGB and then to PIL Image
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = im.fromarray(cv_image_rgb)

            # Extract and convert timestamp
            current_time = msg.header.stamp
            converted_time = current_time.sec + current_time.nanosec * 1e-9

            # Thread-safe access to shared state
            with self.buffer_lock:
                # Initialize start time on first image
                if self.start_time is None:
                    self.start_time = converted_time
                    self.get_logger().info("First image received, starting timing")

                # Buffer size management
                if len(self.image_buffer) >= self.MAX_BUFFER_SIZE:
                    self.get_logger().warn(
                        f"Buffer full ({self.MAX_BUFFER_SIZE}), dropping oldest image")
                    self.image_buffer.pop(0)

                # Store image with timestamp as tuple
                self.image_buffer.append((pil_image, converted_time))

                # Check if segment time has elapsed
                if converted_time - self.start_time >= self.segment_time:
                    before_time = time.time()

                    if self.image_buffer:
                        # Extract images and timestamps separately
                        images_to_process = [img for img, _ in self.image_buffer]
                        timestamps = [ts for _, ts in self.image_buffer]

                        # Create copy of odom buffer for this segment
                        odom_buffer_copy = self.odom_buffer.copy()

                        # Clear buffers for next segment
                        self.image_buffer = []
                        self.odom_buffer = []

                        # Submit to executor with all required data
                        self.vila_executor.submit(
                            self.process_image_segment,
                            images_to_process,
                            timestamps,
                            odom_buffer_copy)

                    after_time = time.time()
                    self.get_logger().info(
                        f"Segment submission time: {after_time - before_time:.4f} seconds")

                    self.start_time = converted_time

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def odom_callback(self, msg):
        """Process incoming odometry messages with thread-safe buffering."""
        try:
            # Extract position
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z

            # Convert quaternion to yaw (theta)
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            theta = math.atan2(2.0 * (qw * qz + qx * qy),
                               1.0 - 2.0 * (qy * qy + qz * qz))

            # Thread-safe buffer access
            with self.buffer_lock:
                # Buffer size management
                if len(self.odom_buffer) >= self.MAX_BUFFER_SIZE:
                    self.odom_buffer.pop(0)

                self.odom_buffer.append((x, y, z, theta))

                # Store as last known pose for fallback
                self.last_pose = (x, y, z, theta)

        except Exception as e:
            self.get_logger().error(f"Error in odom_callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def process_image_segment(self, images, timestamps, odom_buffer):
        try:
            self.get_logger().info(f"Processing segment with {len(images)} images")

            # Build the prompt: [Image1, Image2, ..., ImageN, prompt_text]
            prompt = images.copy()
            prompt.append(self.prompt_text)

            # Generate response using VILA
            self.get_logger().info("Generating caption with VILA...")
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)

            # Log the response
            self.get_logger().info("=" * 60)
            self.get_logger().info("VILA Response:")
            self.get_logger().info(response)
            self.get_logger().info("=" * 60)

            # Calculate averages from synchronized odom or use fallback
            if odom_buffer:
                positions_x = [d[0] for d in odom_buffer]
                positions_y = [d[1] for d in odom_buffer]
                positions_z = [d[2] for d in odom_buffer]
                thetas = [d[3] for d in odom_buffer]

                # Arithmetic mean for positions
                avg_x = sum(positions_x) / len(positions_x)
                avg_y = sum(positions_y) / len(positions_y)
                avg_z = sum(positions_z) / len(positions_z)

                # Circular mean for theta (handle angle wraparound)
                sin_sum = sum(math.sin(t) for t in thetas)
                cos_sum = sum(math.cos(t) for t in thetas)
                avg_theta = math.atan2(sin_sum, cos_sum)

                self.get_logger().info(
                    f"Published CaptionWithPose: {len(images)} images, "
                    f"pos=({avg_x:.2f}, {avg_y:.2f}, {avg_z:.2f}), "
                    f"theta={avg_theta:.2f} rad, "
                    f"{len(odom_buffer)} odom samples")
            elif self.last_pose is not None:
                # Use last known pose as fallback
                avg_x = self.last_pose[0]
                avg_y = self.last_pose[1]
                avg_z = self.last_pose[2]
                avg_theta = self.last_pose[3]

                self.get_logger().warn(
                    f"Published CaptionWithPose (using fallback pose): {len(images)} images, "
                    f"pos=({avg_x:.2f}, {avg_y:.2f}, {avg_z:.2f}), theta={avg_theta:.2f} rad")
            else:
                # No odom data at all - skip publishing
                self.get_logger().error(
                    f"No odom data available for segment "
                    f"(time range: {timestamps[0]:.2f} - {timestamps[-1]:.2f}). Skipping publish.")

            msg = CaptionWithPose()
            msg.caption = response
            msg.position_x = avg_x
            msg.position_y = avg_y
            msg.position_z = avg_z
            msg.theta = avg_theta
            msg.timestamp = self.calculate_median_timestamp(timestamps)
            msg.image_count = len(images)

            self.caption_pose_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error during VILA inference: {e}")
            self.get_logger().error(traceback.format_exc())

    def calculate_median_timestamp(self, timestamps):
        """Calculate median timestamp from list of float timestamps."""
        sorted_times = sorted(timestamps)
        median_float = statistics.median(sorted_times)

        # Convert back to ROS Time
        median_time = Time()
        median_time.sec = int(median_float)
        median_time.nanosec = int((median_float - median_time.sec) * 1e9)

        return median_time

    def destroy_node(self):
        """Clean up resources before node shutdown."""
        try:
            self.get_logger().info("Shutting down ThreadPoolExecutor...")
            self.vila_executor.shutdown(wait=True, cancel_futures=False)
            self.get_logger().info("ThreadPoolExecutor shutdown complete")
        except Exception as e:
            self.get_logger().error(f"Error during executor shutdown: {e}")
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    captioner = Captioner()

    try:
        rclpy.spin(captioner)
    finally:
        captioner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
