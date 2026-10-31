#!/bin/bash
set -euo pipefail
export TMPDIR=/mnt/data/mdevos/conflux/tmp
export CONDA_PKGS_DIRS=/mnt/data/mdevos/conda/pkgs
export PIP_CACHE_DIR=/mnt/data/mdevos/conflux/cache/pip
export XDG_CACHE_HOME=/mnt/data/mdevos/conflux/cache
/home/mdevos/miniconda3/bin/conda create -y -p /mnt/data/mdevos/conda/envs/conflux-gpu -c conda-forge --override-channels python=3.12 pip
/mnt/data/mdevos/conda/envs/conflux-gpu/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
/mnt/data/mdevos/conda/envs/conflux-gpu/bin/python -m pip install numpy==2.5.4 scikit-learn==1.9.1 pandas==3.0.6 matplotlib==3.11.2 dataclasses-json==0.6.7 networkx==3.7 pyzmq==27.2.0 psutil==7.2.2 requests==2.34.2 setproctitle==1.3.8 pytest==9.1.1 pytest-asyncio==1.4.0
/mnt/data/mdevos/conda/envs/conflux-gpu/bin/python -c 'import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'
