from dataclasses import dataclass

from vila_msgs.msg import CaptionWithPose


@dataclass(frozen=True, slots=True)
class CaptionData:
    """Immutable caption data extracted from ROS message."""

    caption: str
    position_x: float
    position_y: float
    position_z: float
    theta: float
    timestamp_sec: int
    timestamp_nanosec: int
    image_count: int

    @classmethod
    def from_ros_msg(cls, msg: CaptionWithPose) -> 'CaptionData':
        """Create from ROS CaptionWithPose message."""
        return cls(
            caption=msg.caption,
            position_x=float(msg.position_x),
            position_y=float(msg.position_y),
            position_z=float(msg.position_z),
            theta=float(msg.theta),
            timestamp_sec=int(msg.timestamp.sec),
            timestamp_nanosec=int(msg.timestamp.nanosec),
            image_count=int(msg.image_count)
        )

    def __post_init__(self) -> None:
        if not self.caption.strip():
            raise ValueError("Caption cannot be empty")
        if not -2 * 3.14159 <= self.theta <= 2 * 3.14159:
            raise ValueError(f"Theta {self.theta} outside valid range")
