from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass(frozen=True, slots=True)
class PoseData:

    x: float
    y: float
    z: float
    theta: float  # radians

    @classmethod
    def from_quaternion(
        cls,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> 'PoseData':
        theta = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return cls(x, y, z, theta)

    @classmethod
    def from_odom_buffer(
        cls,
        buffer: List[Tuple[float, float, float, float]],
    ) -> 'PoseData':
        if not buffer:
            raise ValueError("Cannot compute average from empty buffer")

        positions_x = [d[0] for d in buffer]
        positions_y = [d[1] for d in buffer]
        positions_z = [d[2] for d in buffer]
        thetas = [d[3] for d in buffer]

        avg_x = sum(positions_x) / len(positions_x)
        avg_y = sum(positions_y) / len(positions_y)
        avg_z = sum(positions_z) / len(positions_z)

        # Circular mean handles angle wraparound
        sin_sum = sum(math.sin(t) for t in thetas)
        cos_sum = sum(math.cos(t) for t in thetas)
        avg_theta = math.atan2(sin_sum, cos_sum)

        return cls(avg_x, avg_y, avg_z, avg_theta)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.theta)
