import threading
from collections import deque
from typing import List, Optional, Tuple

from nav_msgs.msg import Odometry

from ..models.pose_data import PoseData
from ..utils.protocols import Logger


class OdomService:

    def __init__(self, max_buffer_size: int, logger: Logger) -> None:
        self._logger = logger
        self._buffer: deque = deque(maxlen=max_buffer_size)
        self._last_pose: Optional[PoseData] = None
        self._lock = threading.Lock()

    def process_odom(self, msg: Odometry) -> PoseData:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        return PoseData.from_quaternion(x, y, z, qx, qy, qz, qw)

    def add_to_buffer(self, pose: PoseData) -> None:
        with self._lock:
            self._buffer.append(pose.as_tuple())
            self._last_pose = pose

    def flush_buffer(self) -> List[Tuple[float, float, float, float]]:
        with self._lock:
            buffer_copy = list(self._buffer)
            self._buffer.clear()
            return buffer_copy

    @property
    def last_pose(self) -> Optional[PoseData]:
        with self._lock:
            return self._last_pose
