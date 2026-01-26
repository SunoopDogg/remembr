import statistics
from typing import List, Tuple, Optional

from builtin_interfaces.msg import Time

from vila_msgs.msg import CaptionWithPose

from ..models.pose_data import PoseData
from ..utils.timestamp_utils import float_to_ros_time


class PoseService:
    """Pose calculation and message building service."""

    @staticmethod
    def calculate_average_pose(
        odom_buffer: List[Tuple[float, float, float, float]],
    ) -> PoseData:
        """Calculate average pose from odometry buffer."""
        return PoseData.from_odom_buffer(odom_buffer)

    @staticmethod
    def calculate_median_timestamp(timestamps: List[float]) -> Time:
        """Calculate median timestamp from list of float timestamps."""
        sorted_times = sorted(timestamps)
        median_float = statistics.median(sorted_times)
        return float_to_ros_time(median_float)

    @staticmethod
    def build_message(
        caption: str,
        pose: PoseData,
        timestamp: Time,
        image_count: int,
        duration: float = 0.0,
    ) -> CaptionWithPose:
        """Build CaptionWithPose message from components."""
        msg = CaptionWithPose()
        msg.caption = caption
        msg.position_x = pose.x
        msg.position_y = pose.y
        msg.position_z = pose.z
        msg.theta = pose.theta
        msg.timestamp = timestamp
        msg.image_count = image_count
        msg.duration = duration
        return msg

    @staticmethod
    def resolve_pose(
        odom_buffer: List[Tuple[float, float, float, float]],
        fallback_pose: Optional[PoseData],
    ) -> Tuple[Optional[PoseData], bool]:
        """Resolve pose from buffer or fallback.

        Returns:
            Tuple of (pose, is_fallback) where is_fallback indicates if fallback was used.
            Returns (None, False) if no pose available.
        """
        if odom_buffer:
            return PoseData.from_odom_buffer(odom_buffer), False
        elif fallback_pose is not None:
            return fallback_pose, True
        else:
            return None, False
