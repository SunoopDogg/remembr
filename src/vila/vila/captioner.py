import time
import traceback
import concurrent.futures
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry

from .config import CaptionerConfig
from .models import ImageSegment
from .services import VilaService, ImageService, OdomService, PoseService


class Captioner(Node):
    """ROS2 node for VILA image captioning with pose."""

    def __init__(self) -> None:
        super().__init__('captioner')

        # Load configuration from ROS2 parameters
        self._config = CaptionerConfig.from_ros_node(self)

        # Initialize services
        self._vila_service = VilaService(
            self._config.model_path,
            self._config.conv_mode,
            self.get_logger(),
        )
        self._image_service = ImageService(
            self._config.max_buffer_size,
            self.get_logger(),
        )
        self._odom_service = OdomService(
            self._config.max_buffer_size,
            self.get_logger(),
        )

        # Setup ROS2 interfaces
        self._setup_subscribers()
        self._setup_publishers()

        # Load VILA model
        self._vila_service.load_model()

        # Thread pool for async processing
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        # Timing state for segment tracking
        self._start_time: Optional[float] = None

    def _setup_subscribers(self) -> None:
        """Configure ROS2 subscribers."""
        self._image_subscriber = self.create_subscription(
            CompressedImage,
            self._config.image_topic,
            self._image_callback,
            10,
        )
        self._odom_subscriber = self.create_subscription(
            Odometry,
            self._config.odom_topic,
            self._odom_callback,
            10,
        )

    def _setup_publishers(self) -> None:
        """Configure ROS2 publishers."""
        from vila_msgs.msg import CaptionWithPose

        self._caption_pose_pub = self.create_publisher(
            CaptionWithPose,
            self._config.output_topic,
            10,
        )
        self.get_logger().info(
            f"CaptionWithPose publisher created on {self._config.output_topic}"
        )

    def _image_callback(self, msg: CompressedImage) -> None:
        """Process incoming image messages."""
        try:
            # Convert ROS image to PIL
            pil_image = self._image_service.convert_ros_to_pil(msg)
            current_time = self._image_service.get_timestamp(msg)

            # Initialize start time on first image
            if self._start_time is None:
                self._start_time = current_time
                self.get_logger().info("First image received, starting timing")

            # Buffer size management with warning
            if self._image_service.add_to_buffer(pil_image, current_time):
                self.get_logger().warn(
                    f"Buffer full ({self._config.max_buffer_size}), dropping oldest image"
                )

            # Check if segment time has elapsed
            if current_time - self._start_time >= self._config.segment_time:
                self._submit_segment_for_processing()
                self._start_time = current_time

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def _odom_callback(self, msg: Odometry) -> None:
        """Process incoming odometry messages."""
        try:
            pose = self._odom_service.process_odom(msg)
            self._odom_service.add_to_buffer(pose)
        except Exception as e:
            self.get_logger().error(f"Error in odom_callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def _submit_segment_for_processing(self) -> None:
        """Submit current buffers for async processing."""
        before_time = time.time()

        if self._image_service.is_empty:
            return

        # Flush buffers
        images, timestamps = self._image_service.flush_buffer()
        odom_buffer = self._odom_service.flush_buffer()

        # Create immutable segment
        segment = ImageSegment.from_lists(images, timestamps)

        # Submit to executor
        self._executor.submit(
            self._process_segment,
            segment,
            odom_buffer,
        )

        after_time = time.time()
        self.get_logger().info(
            f"Segment submission time: {after_time - before_time:.4f} seconds"
        )

    def _process_segment(
        self,
        segment: ImageSegment,
        odom_buffer: List[Tuple[float, float, float, float]],
    ) -> None:
        """Process image segment with VILA inference."""
        try:
            self.get_logger().info(
                f"Processing segment with {segment.image_count} images"
            )

            # Generate caption
            self.get_logger().info("Generating caption with VILA...")
            caption = self._vila_service.generate_caption(
                list(segment.images),
                self._config.prompt_text,
                self._config.temperature,
                self._config.max_new_tokens,
            )

            # Log caption preview
            preview = caption[:100] + "..." if len(caption) > 100 else caption
            self.get_logger().info(f"Caption generated: {preview}")
            self.get_logger().debug(f"Full caption: {caption}")

            # Resolve pose from buffer or fallback
            pose, is_fallback = PoseService.resolve_pose(
                odom_buffer,
                self._odom_service.last_pose,
            )

            if pose is None:
                self.get_logger().error(
                    f"No odom data available for segment "
                    f"(time range: {segment.time_range[0]:.2f} - "
                    f"{segment.time_range[1]:.2f}). Skipping publish."
                )
                return

            # Log pose info
            if is_fallback:
                self.get_logger().warn(
                    f"Published CaptionWithPose (using fallback pose): "
                    f"{segment.image_count} images, "
                    f"pos=({pose.x:.2f}, {pose.y:.2f}, {pose.z:.2f}), "
                    f"theta={pose.theta:.2f} rad"
                )
            else:
                self.get_logger().info(
                    f"Published CaptionWithPose: {segment.image_count} images, "
                    f"pos=({pose.x:.2f}, {pose.y:.2f}, {pose.z:.2f}), "
                    f"theta={pose.theta:.2f} rad, {len(odom_buffer)} odom samples"
                )

            # Build and publish message
            timestamp = PoseService.calculate_median_timestamp(
                list(segment.timestamps)
            )
            msg = PoseService.build_message(
                caption,
                pose,
                timestamp,
                segment.image_count,
            )
            self._caption_pose_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error during VILA inference: {e}")
            self.get_logger().error(traceback.format_exc())

    def destroy_node(self) -> None:
        """Clean up resources before node shutdown."""
        try:
            self.get_logger().info("Shutting down ThreadPoolExecutor...")
            self._executor.shutdown(wait=True, cancel_futures=False)
            self.get_logger().info("ThreadPoolExecutor shutdown complete")
        except Exception as e:
            self.get_logger().error(f"Error during executor shutdown: {e}")
        finally:
            super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    captioner = Captioner()

    try:
        rclpy.spin(captioner)
    except KeyboardInterrupt:
        captioner.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        captioner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
