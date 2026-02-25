"""Launch file for waypoint patrol with rosbag recording."""

import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    EmitEvent,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for waypoint patrol with rosbag recording."""
    # Launch configurations
    auto_start = LaunchConfiguration('auto_start')
    use_sim_time = LaunchConfiguration('use_sim_time')
    record_bag = LaunchConfiguration('record_bag')
    bag_path = LaunchConfiguration('bag_path')
    shutdown_on_complete = LaunchConfiguration('shutdown_on_complete')
    dwell_time = LaunchConfiguration('dwell_time')

    # Default bag path with timestamp (use source directory, not install space)
    default_bag_path = os.path.join(
        os.path.expanduser('~'),
        'remembr', 'src', 'tb3_nav_bringup', 'rosbag',
        f'patrol_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )

    # Declare launch arguments
    declare_auto_start_cmd = DeclareLaunchArgument(
        'auto_start',
        default_value='true',
        description='Auto-start patrol on node launch'
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )

    declare_record_bag_cmd = DeclareLaunchArgument(
        'record_bag',
        default_value='true',
        description='Whether to record rosbag'
    )

    declare_bag_path_cmd = DeclareLaunchArgument(
        'bag_path',
        default_value=default_bag_path,
        description='Path to save rosbag'
    )

    declare_shutdown_on_complete_cmd = DeclareLaunchArgument(
        'shutdown_on_complete',
        default_value='true',
        description='Shutdown node (and rosbag) when patrol completes'
    )

    declare_dwell_time_cmd = DeclareLaunchArgument(
        'dwell_time',
        default_value='5.0',
        description='Time to dwell at each waypoint for object observation (seconds)'
    )

    # Waypoint patrol node
    waypoint_patrol_node = Node(
        package='tb3_nav_bringup',
        executable='waypoint_patrol',
        name='waypoint_patrol_node',
        output='screen',
        parameters=[{
            'auto_start': auto_start,
            'use_sim_time': use_sim_time,
            'shutdown_on_complete': shutdown_on_complete,
            'dwell_time': dwell_time,
        }]
    )

    # Rosbag recording process
    # Topics to record for VILA captioner (minimal data for memory system)
    topics_to_record = [
        # Camera topics (different names depending on simulation setup)
        '/camera/camera/color/image_raw/compressed',  # RealSense style
        '/camera/image_raw/compressed',               # Generic compressed
        '/camera/image_raw',                          # Raw image fallback
        # Odometry
        '/odom',                                      # Pose tracking
    ]

    rosbag_record_cmd = ExecuteProcess(
        condition=IfCondition(record_bag),
        cmd=[
            'ros2', 'bag', 'record',
            '-o', bag_path,
            '--use-sim-time',
        ] + topics_to_record,
        output='screen'
    )

    # Event handler: shutdown everything when patrol node exits
    shutdown_on_patrol_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=waypoint_patrol_node,
            on_exit=[EmitEvent(event=Shutdown(reason='Patrol complete'))]
        )
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_auto_start_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_record_bag_cmd)
    ld.add_action(declare_bag_path_cmd)
    ld.add_action(declare_shutdown_on_complete_cmd)
    ld.add_action(declare_dwell_time_cmd)

    # Add nodes and processes
    ld.add_action(waypoint_patrol_node)
    ld.add_action(rosbag_record_cmd)

    # Add event handlers
    ld.add_action(shutdown_on_patrol_exit)

    return ld
