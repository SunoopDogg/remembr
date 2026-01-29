import threading
from typing import List, Tuple

import cv2
from cv_bridge import CvBridge
from PIL import Image

from sensor_msgs.msg import CompressedImage, Image as RosImage

from ..utils.protocols import Logger
from ..utils.timestamp_utils import ros_time_to_float


class ImageService:
    """Image conversion and buffering service."""

    def __init__(self, max_buffer_size: int, logger: Logger) -> None:
        self._max_buffer_size = max_buffer_size
        self._logger = logger
        self._bridge = CvBridge()
        self._buffer: List[Tuple[Image.Image, float]] = []
        self._lock = threading.Lock()

    def convert_compressed_to_pil(self, msg: CompressedImage) -> Image.Image:
        """Convert ROS CompressedImage to PIL Image."""
        cv_image = self._bridge.compressed_imgmsg_to_cv2(msg)
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_image_rgb)

    def convert_raw_to_pil(self, msg: RosImage) -> Image.Image:
        """Convert ROS Image to PIL Image."""
        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_image_rgb)

    def get_compressed_timestamp(self, msg: CompressedImage) -> float:
        """Extract timestamp from ROS CompressedImage message as float."""
        return ros_time_to_float(msg.header.stamp)

    def get_raw_timestamp(self, msg: RosImage) -> float:
        """Extract timestamp from ROS Image message as float."""
        return ros_time_to_float(msg.header.stamp)

    def add_to_buffer(self, image: Image.Image, timestamp: float) -> bool:
        """Add image to buffer. Returns True if buffer was full and oldest dropped."""
        with self._lock:
            was_full = False
            if len(self._buffer) >= self._max_buffer_size:
                self._buffer.pop(0)
                was_full = True

            self._buffer.append((image, timestamp))
            return was_full

    def flush_buffer(self) -> Tuple[List[Image.Image], List[float]]:
        """Flush buffer and return all images with timestamps."""
        with self._lock:
            if not self._buffer:
                return [], []
            images, timestamps = zip(*self._buffer)
            self._buffer.clear()
            return list(images), list(timestamps)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0
