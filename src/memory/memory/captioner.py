import time
import traceback
import concurrent.futures
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry

from .config import CaptionerConfig
from .models import ImageSegment
from .services import GemmaService, ImageService, OdomService, PoseService


class Captioner(Node):

    def __init__(self) -> None:
        super().__init__('captioner')

        self._config = CaptionerConfig.from_ros_node(self)

        self._gemma_service = GemmaService(
            self._config.model_name,
            self._config.vlm_base_url,
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

        self._setup_subscribers()
        self._setup_publishers()

        self._gemma_service.load_model()

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        self._segment_window_start: Optional[float] = None

    def _log_exception(self, context: str, exc: Exception) -> None:
        self.get_logger().error(f"Error in {context}: {exc}")
        self.get_logger().error(traceback.format_exc())

    def _setup_subscribers(self) -> None:
        self._compressed_image_subscriber = self.create_subscription(
            CompressedImage,
            self._config.compressed_image_topic,
            self._compressed_image_callback,
            10,
        )
        self._raw_image_subscriber = self.create_subscription(
            Image,
            self._config.raw_image_topic,
            self._raw_image_callback,
            10,
        )
        self._odom_subscriber = self.create_subscription(
            Odometry,
            self._config.odom_topic,
            self._odom_callback,
            10,
        )

    def _setup_publishers(self) -> None:
        from memory_msgs.msg import CaptionWithPose

        self._caption_pose_pub = self.create_publisher(
            CaptionWithPose,
            self._config.output_topic,
            10,
        )
        self.get_logger().info(
            f"CaptionWithPose publisher created on {self._config.output_topic}"
        )

    def _compressed_image_callback(self, msg: CompressedImage) -> None:
        try:
            pil_image = self._image_service.convert_compressed_to_pil(msg)
            current_time = self._image_service.get_timestamp(msg)
            self._process_image(pil_image, current_time)
        except Exception as e:
            self._log_exception("compressed_image_callback", e)

    def _raw_image_callback(self, msg: Image) -> None:
        try:
            pil_image = self._image_service.convert_raw_to_pil(msg)
            current_time = self._image_service.get_timestamp(msg)
            self._process_image(pil_image, current_time)
        except Exception as e:
            self._log_exception("raw_image_callback", e)

    def _process_image(self, pil_image, current_time: float) -> None:
        if self._segment_window_start is None:
            self._segment_window_start = current_time
            self.get_logger().info("First image received, starting timing")

        if self._image_service.add_to_buffer(pil_image, current_time):
            self.get_logger().warn(
                f"Buffer full ({self._config.max_buffer_size}), dropping oldest image"
            )

        if current_time - self._segment_window_start >= self._config.segment_time:
            self._submit_segment_for_processing()
            self._segment_window_start = current_time

    def _odom_callback(self, msg: Odometry) -> None:
        try:
            pose = self._odom_service.process_odom(msg)
            self._odom_service.add_to_buffer(pose)
        except Exception as e:
            self._log_exception("odom_callback", e)

    def _submit_segment_for_processing(self) -> None:
        if self._image_service.is_empty:
            return

        before_time = time.time()

        images, timestamps = self._image_service.flush_buffer()
        odom_buffer = self._odom_service.flush_buffer()

        segment = ImageSegment.from_lists(images, timestamps)

        self._executor.submit(
            self._process_segment,
            segment,
            odom_buffer,
        )

        after_time = time.time()
        self.get_logger().debug(
            f"Segment submission time: {after_time - before_time:.4f} seconds"
        )

    def _process_segment(
        self,
        segment: ImageSegment,
        odom_buffer: List[Tuple[float, float, float, float]],
    ) -> None:
        try:
            self.get_logger().info(
                f"Processing segment with {segment.image_count} images"
            )

            caption = self._gemma_service.generate_caption(
                segment.images,
                self._config.prompt_text,
                self._config.temperature,
                self._config.max_tokens,
            )

            preview = caption[:100] + "..." if len(caption) > 100 else caption
            self.get_logger().info(f"Caption generated: {preview}")
            self.get_logger().debug(f"Full caption: {caption}")

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

            timestamp = PoseService.calculate_median_timestamp(segment.timestamps)
            msg = PoseService.build_message(
                caption,
                pose,
                timestamp,
            )
            self._caption_pose_pub.publish(msg)

        except Exception as e:
            self._log_exception("Gemma inference", e)

    def destroy_node(self) -> None:
        try:
            self.get_logger().info("Shutting down ThreadPoolExecutor...")
            self._executor.shutdown(wait=True, cancel_futures=False)
            self.get_logger().info("ThreadPoolExecutor shutdown complete")
        except Exception as e:
            self._log_exception("executor shutdown", e)
        finally:
            self._gemma_service.cleanup()
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
