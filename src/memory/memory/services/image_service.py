import threading
from typing import List, Tuple

import cv2
from cv_bridge import CvBridge
from PIL import Image

from sensor_msgs.msg import CompressedImage, Image as RosImage

from ..utils.protocols import Logger
from ..utils.timestamp_utils import ros_time_to_float


class ImageService:

    def __init__(self, max_buffer_size: int, logger: Logger) -> None:
        self._max_buffer_size = max_buffer_size
        self._logger = logger
        self._bridge = CvBridge()
        self._buffer: List[Tuple[Image.Image, float]] = []
        self._lock = threading.Lock()

    def _cv_to_pil(self, cv_image) -> Image.Image:
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_image_rgb)

    def convert_compressed_to_pil(self, msg: CompressedImage) -> Image.Image:
        return self._cv_to_pil(self._bridge.compressed_imgmsg_to_cv2(msg))

    def convert_raw_to_pil(self, msg: RosImage) -> Image.Image:
        return self._cv_to_pil(self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))

    def get_timestamp(self, msg) -> float:
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
        with self._lock:
            if not self._buffer:
                return [], []
            images, timestamps = zip(*self._buffer)
            self._buffer.clear()
            return list(images), list(timestamps)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0
