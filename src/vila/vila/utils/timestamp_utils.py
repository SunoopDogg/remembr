from builtin_interfaces.msg import Time


def ros_time_to_float(stamp: Time) -> float:
    """Convert ROS Time message to float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def float_to_ros_time(seconds: float) -> Time:
    """Convert float seconds to ROS Time message."""
    time = Time()
    time.sec = int(seconds)
    time.nanosec = int((seconds - time.sec) * 1e9)
    return time
