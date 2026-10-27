# NVIDIA Jetson TX2 Configuration

## Hardware
- GPU: NVIDIA Pascal Architecture GPU
- CUDA Cores: 256
- GPU Frequency: Up to 1.3 GHz
- GPU Performance: 1.33 TFLOPS
- CPU:
  - Dual-core NVIDIA Denver 2 64-bit CPU
  - Quad-core ARM Cortex-A57 MPCore
- CPU Frequency: Up to 2.0 GHz

## Memory and Storage
- RAM: 8 GB LPDDR4
- Memory Bandwidth: 59.7 GB/s
- Storage: 32 GB eMMC 5.1

## Software Environment
- JetPack: 4.6.6
- TensorFlow: 2.6.2

## Purpose
Used for edge deployment benchmarking:
- Model inference evaluation
- Footprint size analysis
- Inference latency measurement
- Accuracy comparison after optimization

## Jetson TX2 dependencies
numpy==1.19.4
future==0.18.2
mock==3.0.5
keras_preprocessing==1.1.2
keras_applications==1.0.8
gast==0.4.0
protobuf==3.19.6
pybind11
cython==0.29.37
pkgconfig