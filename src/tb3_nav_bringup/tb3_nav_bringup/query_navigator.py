"""Query Navigator - Sends navigation goals based on agent responses."""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from agent_msgs.srv import Query


class QueryNavigator(Node):
    """Node that queries the agent and navigates to returned positions."""

    def __init__(self) -> None:
        super().__init__('query_navigator')

        # Parameters
        self.declare_parameter('goal_frame', 'map')
        self.declare_parameter('agent_service', '/remembr_agent/query')

        self._goal_frame = self.get_parameter('goal_frame').value
        self._agent_service = self.get_parameter('agent_service').value

        # Service client for agent queries
        self._query_client = self.create_client(Query, self._agent_service)

        # Publisher for navigation goals (Nav2)
        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Subscriber for query commands
        self._query_sub = self.create_subscription(
            String,
            '~/query',
            self._query_callback,
            10
        )

        self.get_logger().info(f'QueryNavigator initialized')
        self.get_logger().info(f'  Agent service: {self._agent_service}')
        self.get_logger().info(f'  Query topic: {self.get_name()}/query')

    def _query_callback(self, msg: String) -> None:
        """Handle incoming query and navigate if position returned."""
        question = msg.data.strip()
        if not question:
            self.get_logger().warning('Empty query received')
            return

        self.get_logger().info(f'Query: "{question[:50]}..."')

        # Wait for service
        if not self._query_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Agent service not available: {self._agent_service}')
            return

        # Call agent service
        request = Query.Request()
        request.question = question

        future = self._query_client.call_async(request)
        future.add_done_callback(lambda f: self._handle_response(f, question))

    def _handle_response(self, future, question: str) -> None:
        """Process agent response and send navigation goal if applicable."""
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return

        if not response.success:
            self.get_logger().warning(f'Query failed: {response.error}')
            return

        self.get_logger().info(f'Response type: {response.type}')
        self.get_logger().info(f'Response text: {response.text[:100] if response.text else "N/A"}...')

        # Check if position is available
        if response.position and len(response.position) >= 3:
            self._send_navigation_goal(
                position=response.position,
                orientation=response.orientation
            )
        else:
            self.get_logger().info('No position in response - skipping navigation')

    def _send_navigation_goal(
        self,
        position: list,
        orientation: float
    ) -> None:
        """Send navigation goal to Nav2."""
        goal = PoseStamped()
        goal.header.frame_id = self._goal_frame
        goal.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal.pose.position.x = float(position[0])
        goal.pose.position.y = float(position[1])
        goal.pose.position.z = float(position[2]) if len(position) > 2 else 0.0

        # Orientation (euler Z to quaternion)
        yaw = float(orientation) if orientation else 0.0
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        self._goal_pub.publish(goal)

        self.get_logger().info(
            f'Navigation goal sent: '
            f'pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), '
            f'yaw={yaw:.2f} rad'
        )


def main(args=None) -> None:
    """Main entry point."""
    rclpy.init(args=args)

    node = QueryNavigator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
