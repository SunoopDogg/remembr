import statistics
from typing import List, Optional, Tuple

from builtin_interfaces.msg import Time
from sensor_msgs.msg import CompressedImage as SensorCompressedImage
from memory_msgs.msg import CaptionWithPose

from ..models.pose_data import PoseData
from ..utils.timestamp_utils import float_to_ros_time


class PoseService:

    @staticmethod
    def calculate_median_timestamp(timestamps: List[float]) -> Time:
        median_float = statistics.median(timestamps)
        return float_to_ros_time(median_float)

    @staticmethod
    def build_message(
        caption: str,
        pose: PoseData,
        timestamp: Time,
        images: Optional[List[SensorCompressedImage]] = None,
    ) -> CaptionWithPose:
        msg = CaptionWithPose()
        msg.caption = caption
        msg.position_x = pose.x
        msg.position_y = pose.y
        msg.position_z = pose.z
        msg.theta = pose.theta
        msg.timestamp = timestamp
        if images:
            msg.images = images
        return msg

    @staticmethod
    def resolve_pose(
        odom_buffer: List[Tuple[float, float, float, float]],
        fallback_pose: Optional[PoseData],
    ) -> Tuple[Optional[PoseData], bool]:
        if odom_buffer:
            return PoseData.from_odom_buffer(odom_buffer), False
        elif fallback_pose is not None:
            return fallback_pose, True
        else:
            return None, False
