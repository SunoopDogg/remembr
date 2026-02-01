#!/usr/bin/env python3
"""
Waypoint patrol node for TurtleBot3 navigation.

This node provides a service-triggered waypoint patrol system that navigates
through predefined waypoints in small_house.world and returns to the start point.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import math

from tb3_nav_bringup.patrol_waypoints import WAYPOINTS, START_POINT


class PatrolState:
    """Patrol state enumeration."""
    IDLE = "IDLE"
    PATROLLING = "PATROLLING"
    RETURNING = "RETURNING"


class WaypointPatrolNode(Node):
    """
    Waypoint patrol node using Nav2 BasicNavigator.

    Provides:
    - /start_patrol service to trigger patrol
    - /patrol_status topic publishing current state and waypoint
    """

    def __init__(self):
        super().__init__('waypoint_patrol_node')

        # Declare parameters
        self.declare_parameter('auto_start', True)
        self.declare_parameter('shutdown_on_complete', False)
        # Initial pose for AMCL (should match Gazebo spawn position in small_house.world)
        self.declare_parameter('initial_pose_x', -3.5)
        self.declare_parameter('initial_pose_y', -4.5)
        self.declare_parameter('initial_pose_yaw', 1.58)

        self.auto_start = self.get_parameter('auto_start').value
        self.shutdown_on_complete = self.get_parameter('shutdown_on_complete').value
        self.initial_pose_x = self.get_parameter('initial_pose_x').value
        self.initial_pose_y = self.get_parameter('initial_pose_y').value
        self.initial_pose_yaw = self.get_parameter('initial_pose_yaw').value

        # Initialize BasicNavigator
        self.navigator = BasicNavigator()

        # State management
        self.state = PatrolState.IDLE
        self.current_waypoint_idx = 0
        self.is_patrolling = False

        # Service to start patrol (optional, for manual trigger)
        self.start_patrol_service = self.create_service(
            Trigger,
            '/start_patrol',
            self.start_patrol_callback
        )

        # Publisher for patrol status
        self.status_publisher = self.create_publisher(
            String,
            '/patrol_status',
            10
        )

        # Timer for status updates (1 Hz)
        self.status_timer = self.create_timer(1.0, self.publish_status)

        self.get_logger().info('Waypoint Patrol Node initialized')
        self.get_logger().info(f'Auto start: {self.auto_start}')
        self.get_logger().info(f'Total waypoints: {len(WAYPOINTS)}')

        # Auto start patrol if enabled
        if self.auto_start:
            self.get_logger().info('Auto-starting patrol...')
            self.auto_start_timer = self.create_timer(1.0, self.auto_start_patrol)

    def auto_start_patrol(self):
        """Auto-start patrol after node initialization."""
        # Cancel this one-shot timer
        self.auto_start_timer.cancel()

        # Set initial pose for AMCL localization
        self.get_logger().info('Setting initial pose for AMCL...')
        initial_pose = self.create_pose_stamped({
            'name': 'initial',
            'x': self.initial_pose_x,
            'y': self.initial_pose_y,
            'yaw': self.initial_pose_yaw
        })
        self.navigator.setInitialPose(initial_pose)

        # Wait for navigation to be ready
        self.get_logger().info('Waiting for Nav2 to become active...')
        self.navigator.waitUntilNav2Active()

        # Start patrol
        self.state = PatrolState.PATROLLING
        self.current_waypoint_idx = 0
        self.is_patrolling = True

        # Navigate to first waypoint
        self.navigate_to_current_waypoint()

        # Start patrol loop timer
        self.patrol_timer = self.create_timer(0.5, self.patrol_loop)

        self.get_logger().info(f'Patrol started with {len(WAYPOINTS)} waypoints')

    def yaw_to_quaternion(self, yaw):
        """
        Convert yaw angle to quaternion.

        Args:
            yaw: Yaw angle in radians

        Returns:
            dict: Quaternion components {x, y, z, w}
        """
        half_yaw = yaw / 2.0
        return {
            'x': 0.0,
            'y': 0.0,
            'z': math.sin(half_yaw),
            'w': math.cos(half_yaw)
        }

    def create_pose_stamped(self, waypoint):
        """
        Create a PoseStamped message from waypoint data.

        Args:
            waypoint: Dictionary with 'name', 'x', 'y', 'yaw' keys

        Returns:
            PoseStamped: Nav2 compatible pose message
        """
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()

        pose.pose.position.x = waypoint['x']
        pose.pose.position.y = waypoint['y']
        pose.pose.position.z = 0.0

        quat = self.yaw_to_quaternion(waypoint['yaw'])
        pose.pose.orientation.x = quat['x']
        pose.pose.orientation.y = quat['y']
        pose.pose.orientation.z = quat['z']
        pose.pose.orientation.w = quat['w']

        return pose

    def start_patrol_callback(self, request, response):
        """
        Service callback to start patrol.

        Args:
            request: Trigger request (empty)
            response: Trigger response

        Returns:
            Trigger.Response: Success status and message
        """
        if self.state != PatrolState.IDLE:
            response.success = False
            response.message = f'Cannot start patrol: currently in {self.state} state'
            self.get_logger().warn(response.message)
            return response

        # Set initial pose for AMCL localization
        initial_pose = self.create_pose_stamped({
            'name': 'initial',
            'x': self.initial_pose_x,
            'y': self.initial_pose_y,
            'yaw': self.initial_pose_yaw
        })
        self.navigator.setInitialPose(initial_pose)

        # Wait for navigation to be ready
        self.navigator.waitUntilNav2Active()

        # Start patrol
        self.state = PatrolState.PATROLLING
        self.current_waypoint_idx = 0
        self.is_patrolling = True

        # Navigate to first waypoint
        self.navigate_to_current_waypoint()

        # Start patrol loop timer
        self.patrol_timer = self.create_timer(0.5, self.patrol_loop)

        response.success = True
        response.message = f'Patrol started with {len(WAYPOINTS)} waypoints'
        self.get_logger().info(response.message)

        return response

    def navigate_to_current_waypoint(self):
        """Navigate to the current waypoint in the list."""
        if self.current_waypoint_idx < len(WAYPOINTS):
            waypoint = WAYPOINTS[self.current_waypoint_idx]
            pose = self.create_pose_stamped(waypoint)
            self.navigator.goToPose(pose)

            self.get_logger().info(
                f'Navigating to waypoint {self.current_waypoint_idx + 1}/{len(WAYPOINTS)}: '
                f'{waypoint["name"]} at ({waypoint["x"]:.2f}, {waypoint["y"]:.2f})'
            )

    def navigate_to_start(self):
        """Navigate back to the start point."""
        pose = self.create_pose_stamped(START_POINT)
        self.navigator.goToPose(pose)

        self.get_logger().info(
            f'Returning to start point: {START_POINT["name"]} at '
            f'({START_POINT["x"]:.2f}, {START_POINT["y"]:.2f})'
        )

    def patrol_loop(self):
        """Main patrol loop - check navigation status and advance waypoints."""
        if not self.is_patrolling:
            return

        # Check if current navigation task is complete
        if self.navigator.isTaskComplete():
            result = self.navigator.getResult()

            if result == TaskResult.SUCCEEDED:
                if self.state == PatrolState.PATROLLING:
                    # Successfully reached current waypoint
                    waypoint = WAYPOINTS[self.current_waypoint_idx]
                    self.get_logger().info(f'Reached waypoint: {waypoint["name"]}')

                    # Move to next waypoint
                    self.current_waypoint_idx += 1

                    if self.current_waypoint_idx < len(WAYPOINTS):
                        # More waypoints to visit
                        self.navigate_to_current_waypoint()
                    else:
                        # All waypoints completed, return to start
                        self.get_logger().info('All waypoints completed, returning to start')
                        self.state = PatrolState.RETURNING
                        self.navigate_to_start()

                elif self.state == PatrolState.RETURNING:
                    # Successfully returned to start point
                    self.get_logger().info('Returned to start point, patrol complete')
                    self.state = PatrolState.IDLE
                    self.is_patrolling = False
                    self.patrol_timer.cancel()
                    self._handle_patrol_complete()

            elif result == TaskResult.CANCELED:
                self.get_logger().warn('Navigation was canceled')
                self.state = PatrolState.IDLE
                self.is_patrolling = False
                self.patrol_timer.cancel()

            elif result == TaskResult.FAILED:
                if self.state == PatrolState.PATROLLING:
                    waypoint = WAYPOINTS[self.current_waypoint_idx]
                    self.get_logger().error(f'Failed to reach waypoint: {waypoint["name"]}')
                else:
                    self.get_logger().error('Failed to return to start point')

                self.state = PatrolState.IDLE
                self.is_patrolling = False
                self.patrol_timer.cancel()

    def _handle_patrol_complete(self):
        """Handle patrol completion - optionally shutdown the node."""
        if self.shutdown_on_complete:
            self.get_logger().info('Patrol complete, shutting down node...')
            # Use a short timer to allow final status publish before shutdown
            self.create_timer(1.0, self._shutdown_node)

    def _shutdown_node(self):
        """Shutdown the node gracefully."""
        raise SystemExit(0)

    def publish_status(self):
        """Publish current patrol status."""
        if self.state == PatrolState.IDLE:
            status_msg = f'{self.state}: Waiting for patrol command'

        elif self.state == PatrolState.PATROLLING:
            if self.current_waypoint_idx < len(WAYPOINTS):
                waypoint = WAYPOINTS[self.current_waypoint_idx]
                status_msg = (
                    f'{self.state}: Navigating to {waypoint["name"]} '
                    f'(waypoint {self.current_waypoint_idx + 1}/{len(WAYPOINTS)})'
                )
            else:
                status_msg = f'{self.state}: All waypoints completed'

        elif self.state == PatrolState.RETURNING:
            status_msg = f'{self.state}: Returning to start point'

        else:
            status_msg = f'Unknown state: {self.state}'

        msg = String()
        msg.data = status_msg
        self.status_publisher.publish(msg)


def main(args=None):
    """Main entry point for the waypoint patrol node."""
    rclpy.init(args=args)

    node = WaypointPatrolNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down waypoint patrol node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
