"""
Quick sanity test for the SplitNetwork baseline.

Usage (from repo root):

    python scripts/models/test_split_network.py --device cuda --batch_size 2

This script does not depend on the DRL simulator or accuracy-fitting logic.
It simply instantiates SplitNetwork, feeds a mini-batch (random or from NYUv2),
and prints tensor shapes to ensure the interface is consistent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.split_network import SplitNetwork  # noqa: E402


def load_samples(
    splits_path: Path,
    split: str,
    batch_size: int,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Load a mini-batch of RGB images from splits.json."""
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    entries: List[Dict[str, str]] = splits[split][:batch_size]
    transform = transforms.Compose(
        [
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    images = []
    for entry in entries:
        img = Image.open(entry["rgb"]).convert("RGB")
        images.append(transform(img))
    return torch.stack(images, dim=0)


def build_default_config(output_resolution: Tuple[int, int]) -> Dict:
    """Default SplitNetwork config suitable for quick tests."""
    return {
        "name": "split_resnet34_baseline",
        "architecture": {
            "backbone": "resnet34",
            "pretrained": True,
            "num_classes": 13,
            "output_resolution": list(output_resolution),
            "decoder_channels": [256, 128, 64, 32],
            "task_heads": {
                "seg": {"out_channels": 13, "channels": [256, 128, 64, 32]},
                "depth": {"out_channels": 1, "channels": [128, 64, 32]},
                "normal": {"out_channels": 3, "channels": [128, 64, 32]},
            },
        },
        "input_resolution": [3, output_resolution[0], output_resolution[1]],
    }


def main():
    parser = argparse.ArgumentParser(description="Test SplitNetwork forward pass.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--use_random", action="store_true", help="Use random tensor instead of dataset")
    parser.add_argument("--splits_path", type=str, default="data/processed/nyuv2/splits.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=288)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    resolution = (args.height, args.width)
    config = build_default_config(resolution)

    model = SplitNetwork(config).to(device).eval()

    if args.use_random:
        inputs = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    else:
        splits_path = REPO_ROOT / args.splits_path
        inputs = load_samples(splits_path, args.split, args.batch_size, resolution).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        features = model(inputs, split_point=0)

    print("=== SplitNetwork forward ===")
    for task_name, tensor in outputs.items():
        print(f"{task_name:>6}: {tuple(tensor.shape)}")

    feat = features["features"]
    print("\n=== SplitNetwork split_point=0 ===")
    print(f"features: {tuple(feat.shape)}")


if __name__ == "__main__":
    main()

