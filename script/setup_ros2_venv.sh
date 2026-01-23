uv sync

VENV_DIR="${1:-.venv}"
PTH_FILE="ros2.pth"

PYTHON_VERSION=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages"

cat > "$SITE_PACKAGES/$PTH_FILE" << EOL
/usr/lib/python3/dist-packages
/opt/ros/${ROS_DISTRO}/lib/python${PYTHON_VERSION}/site-packages
/opt/ros/${ROS_DISTRO}/local/lib/python${PYTHON_VERSION}/dist-packages
EOL

source "$VENV_DIR/bin/activate"