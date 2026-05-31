"""
Quick CUDA sanity check.

This script prints Torch/CUDA/cuDNN info and runs a tiny GPU op to confirm
that CUDA is actually usable (not just "installed").

Run:
  python scripts/check_cuda.py
"""

import sys


def main() -> int:
    try:
        import torch
    except Exception as e:
        print(f"[ERROR] Failed to import torch: {e}")
        return 1

    print("=" * 70)
    print("CUDA sanity check")
    print("=" * 70)
    print(f"python: {sys.version.split()[0]}")
    print(f"torch : {torch.__version__}")
    print(f"cuda  : {torch.version.cuda}")
    print(f"cudnn : {torch.backends.cudnn.version()}")
    print(f"is_available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[FAIL] torch.cuda.is_available() is False.")
        print("Possible causes: wrong torch build (CPU-only), driver issue, or CUDA not installed.")
        return 2

    n = torch.cuda.device_count()
    print(f"device_count: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"  [{i}] {props.name} | CC {props.major}.{props.minor} | {total_gb:.2f} GB")

    # Tiny GPU op (keep it small to avoid stressing the system)
    device = torch.device("cuda:0")
    x = torch.randn(512, 512, device=device)
    y = torch.randn(512, 512, device=device)
    z = x @ y
    torch.cuda.synchronize()
    print("\n[OK] Tiny matmul on GPU succeeded.")
    print(f"  result mean: {z.mean().item():.6f}")

    # Optional memory info
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        print(f"  mem_free/total: {free_b / (1024**3):.2f} / {total_b / (1024**3):.2f} GB")
    except Exception:
        pass

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

