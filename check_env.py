import torch
import sys

print("=" * 50)
print("Environment Check")
print("=" * 50)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("WARNING: CUDA not available!")

print("=" * 50)

