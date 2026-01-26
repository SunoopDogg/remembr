apt update

apt install ros-humble-cv-bridge -q -y
apt install zstd -q -y

source ./venv/bin/activate

uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

curl -fsSL https://ollama.com/install.sh | sh

rm -rf /var/lib/apt/lists/*
# git clone https://huggingface.co/Efficient-Large-Model/NVILA-Lite-8B