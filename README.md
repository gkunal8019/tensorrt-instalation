# TensorRT Installation and Docker Guide

## Installation Requirements

**CUDA and PyTorch Dependencies:**
- CUDA 12.x
- PyTorch compatible with CUDA 12
- Python 3.8-3.12[4]

## Installation Steps

```bash
# Update pip and install wheel
python3 -m pip install --upgrade pip
python3 -m pip install wheel

# Install TensorRT
pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.0.1
```

## Docker Usage

```bash
# Pull the Docker image
docker pull kunalgawande/tensor-rt_yolo_robotic:0.1

# Run the container
docker run -it --gpus all kunalgawande/tensor-rt_yolo_robotic:0.1
```

## Features

- TensorRT optimized inference for NVIDIA GPUs[5]
- High-performance C++ library for GPU inference[5]
- Supports model import from:
  - ONNX
  - TensorFlow
  - Caffe

## Key Capabilities

- Graph optimizations and layer fusion
- Mixed precision capabilities
- Optimized kernel selection
- Runtime engine generation for trained networks[5]

## System Requirements

- NVIDIA GPU (Kepler generation or newer)
- Linux operating system
- NVIDIA driver compatible with CUDA 12.x
- Docker with NVIDIA Container Runtime

## Notes

- The Docker image contains all necessary dependencies
- TensorRT provides both C++ and Python APIs
- Supports various precision modes including FP32, FP16, and INT8

## License

This project uses TensorRT which is subject to NVIDIA's EULA[5].

Citations:

[1] https://pypi.nvidia.com

[2] https://pytorch.org/TensorRT/getting_started/installation.html

[3] https://docs.ultralytics.com/guides/nvidia-jetson/

[4] https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html

[5] https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt
