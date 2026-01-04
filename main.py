# uv add torch

import torch
import sys

print("Python version:", sys.version)
print("Executable path:", sys.executable)
print("Platform:", sys.platform)

print()

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU device:", torch.cuda.current_device())
    print("GPU device name:", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A")
    print("GPU device properties:", torch.cuda.get_device_properties(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A")