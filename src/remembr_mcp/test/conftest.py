"""Mock rclpy and ROS2 message types so tests run without a ROS2 installation."""
import sys
from unittest.mock import MagicMock

for mod in [
    'rclpy',
    'rclpy.node',
    'geometry_msgs',
    'geometry_msgs.msg',
    'std_msgs',
    'std_msgs.msg',
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
