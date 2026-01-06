import threading
from typing import List, Tuple, Optional, Protocol

from nav_msgs.msg import Odometry

from ..models.pose_data import PoseData


class Logger(Protocol):
    """Protocol for ROS2-compatible logger."""

    def info(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


class OdomService:
    """Odometry processing and buffering service."""

    def __init__(self, max_buffer_size: int, logger: Logger) -> None:
        self._max_buffer_size = max_buffer_size
        self._logger = logger
        self._buffer: List[Tuple[float, float, float, float]] = []
        self._last_pose: Optional[PoseData] = None
        self._lock = threading.Lock()

    def process_odom(self, msg: Odometry) -> PoseData:
        """Extract pose from Odometry message."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        return PoseData.from_quaternion(x, y, z, qx, qy, qz, qw)

    def add_to_buffer(self, pose: PoseData) -> None:
        """Add pose to buffer with thread-safe access."""
        with self._lock:
            if len(self._buffer) >= self._max_buffer_size:
                self._buffer.pop(0)

            self._buffer.append(pose.as_tuple())
            self._last_pose = pose

    def flush_buffer(self) -> List[Tuple[float, float, float, float]]:
        """Extract and clear buffer contents thread-safely."""
        with self._lock:
            buffer_copy = self._buffer.copy()
            self._buffer.clear()
            return buffer_copy

    def get_buffer_copy(self) -> List[Tuple[float, float, float, float]]:
        """Get a copy of the current buffer without clearing."""
        with self._lock:
            return self._buffer.copy()

    @property
    def last_pose(self) -> Optional[PoseData]:
        """Get last known pose as fallback."""
        with self._lock:
            return self._last_pose

    @property
    def buffer_size(self) -> int:
        """Current buffer size."""
        with self._lock:
            return len(self._buffer)
