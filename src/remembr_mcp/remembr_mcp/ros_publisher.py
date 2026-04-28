import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


class RosPublisher:
    """ROS2 publisher node for goal pose and response topic."""

    def __init__(self) -> None:
        self._node: Node = rclpy.create_node('remembr_mcp')
        self._goal_pub = self._node.create_publisher(PoseStamped, '/goal_pose', 10)
        self._response_pub = self._node.create_publisher(
            String, '/remembr_mcp/response_topic', 10
        )

    def publish(self, result: dict) -> None:
        """Publish result to ROS2 topics."""
        self._publish_response(result)
        position = result.get('position')
        if position and len(position) >= 2:
            self._publish_goal_pose(result)

    def _publish_response(self, result: dict) -> None:
        msg = String()
        msg.data = json.dumps(result, default=str)
        self._response_pub.publish(msg)

    def _publish_goal_pose(self, result: dict) -> None:
        pos = result['position']
        yaw = float(result.get('orientation') or 0.0)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.position.x = float(pos[0])
        goal.pose.position.y = float(pos[1])
        goal.pose.position.z = float(pos[2]) if len(pos) > 2 else 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        self._goal_pub.publish(goal)

    def destroy(self) -> None:
        self._node.destroy_node()
