import time
import threading
from typing import List, Optional, Tuple

import rclpy
import rclpy.executors

from .captioner import Captioner
from .models import ImageSegment


class CaptionerOnce(Captioner):
    """Publishes exactly one CaptionWithPose then signals completion."""

    def __init__(self) -> None:
        super().__init__()
        self._first_image_time: Optional[float] = None
        self._done = threading.Event()

    def _handle_image_msg(self, convert_fn, context: str, msg) -> None:
        if self._first_image_time is None:
            self._first_image_time = time.time()
        super()._handle_image_msg(convert_fn, context, msg)

    def _process_segment(
        self,
        segment: ImageSegment,
        odom_buffer: List[Tuple[float, float, float, float]],
    ) -> None:
        if self._done.is_set():
            return
        super()._process_segment(segment, odom_buffer)
        if self._first_image_time is not None:
            elapsed = time.time() - self._first_image_time
            self.get_logger().info(f"Done. Elapsed: {elapsed:.2f}s (first image -> caption published)")
        self._done.set()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CaptionerOnce()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    while rclpy.ok() and not node._done.is_set():
        executor.spin_once(timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()
