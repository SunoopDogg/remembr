git clone https://huggingface.co/Efficient-Large-Model/NVILA-Lite-8B

source /opt/ros/humble/setup.bash

apt update
apt install ros-humble-cv-bridge -q -y

rm -rf /var/lib/apt/lists/*

source ./venv/bin/activate

uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl