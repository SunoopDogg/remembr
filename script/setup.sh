git clone https://huggingface.co/Efficient-Large-Model/NVILA-Lite-8B

source /opt/ros/humble/setup.bash

source ./venv/bin/activate
uv add flash-attn --no-build-isolation

apt update
apt install ros-humble-cv-bridge -y

rm -rf /var/lib/apt/lists/*