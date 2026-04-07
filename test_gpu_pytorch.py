#! /usr/bin/python3

## This script tests GPU allocation and cuDNN functionality using PyTorch.
## It retrieves OS and GPU information, performs a simple matrix multiplication on the GPU, and tests a convolutional layer using cuDNN.
## Note: Ensure that PyTorch is installed with CUDA support and that a compatible GPU is available for testing.
## Usage:
##     python test_gpu_pytorch.py
##

import platform
import torch
import torch.nn as nn

def get_os_info():
    return platform.platform()

def get_gpu_info():
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        gpu_info.append({
            'name': torch.cuda.get_device_name(device),
            'memory_total': torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        })
    return gpu_info

def test_gpu_allocation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)
    c = torch.matmul(a, b)
    return c.cpu().numpy()

def test_cudnn():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layer = nn.Conv2d(3, 2, 3).to(device)
    x = torch.ones(1, 3, 5, 5).to(device)
    y = layer(x)
    return y.cpu().detach().numpy()

if __name__ == "__main__":
    print(f"OS Info: {get_os_info()}")
    gpus = get_gpu_info()
    print("GPUs:")
    for gpu in gpus:
        print(f"Name: {gpu['name']}, Memory: {gpu['memory_total']} MB")

    if gpus:
        print("Testing GPU Allocation...")
        result = test_gpu_allocation()
        print("Allocation Test Result:", result)

        print("Testing cuDNN...")
        result = test_cudnn()
        print("cuDNN Test Result:", result)

    else:
        print("No GPUs found.")



