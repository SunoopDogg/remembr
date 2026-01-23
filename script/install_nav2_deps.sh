apt update

apt install -q -y ros-${ROS_DISTRO}-rviz2
apt install -q -y ros-${ROS_DISTRO}-turtlebot3-gazebo

git submodule add -b ${ROS_DISTRO} https://github.com/ros-navigation/navigation2.git src/navigation2

source /opt/ros/${ROS_DISTRO}/setup.bash

rosdep install -y \
  --from-paths ./src \
  --ignore-src

rm -rf /var/lib/apt/lists/*