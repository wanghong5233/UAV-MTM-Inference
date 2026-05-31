"""
Preprocess NYUv2 dataset for multi-task learning.

The dataset already has train/test split. This script:
1. Splits train into train/val (90/10)
2. Generates splits.json with sample paths
3. Creates directory structure for processed data

Note: Resizing is done on-the-fly by DataLoader to save disk space.

Usage:
    python scripts/data/1_preprocess.py
    python scripts/data/1_preprocess.py --input_dir data/raw/nyu_v2/data
"""

import argparse
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_csv_pairs(csv_path):
    """Load image-label pairs from CSV file."""
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                rgb_path, label_path = row
                pairs.append({
                    'rgb': rgb_path,
                    'label': label_path
                })
    return pairs


def split_train_val(train_pairs, val_ratio=0.1, seed=42):
    """Split training data into train and validation sets."""
    import random
    random.seed(seed)
    
    # Shuffle
    pairs = train_pairs.copy()
    random.shuffle(pairs)
    
    # Split
    n_val = int(len(pairs) * val_ratio)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    
    return train_pairs, val_pairs


def main():
    parser = argparse.ArgumentParser(description='Preprocess NYUv2 dataset')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/raw/nyu_v2',
        help='Root directory of extracted NYUv2 dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/nyuv2',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1 = 10%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*60)
    print("NYUv2 Dataset Preprocessing")
    print("="*60)
    
    # Check input directory
    if not input_dir.exists():
        print(f"\n✗ Error: Input directory not found: {input_dir}")
        print("\nMake sure you've extracted the dataset:")
        print("  Expected structure: data/raw/nyu_v2/")
        print("    ├─ data/")
        print("    │   ├─ nyu2_train/")
        print("    │   ├─ nyu2_test/")
        print("    │   ├─ nyu2_train.csv")
        print("    │   └─ nyu2_test.csv")
        sys.exit(1)
    
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train and test CSV files
    train_csv = input_dir / "data" / "nyu2_train.csv"
    test_csv = input_dir / "data" / "nyu2_test.csv"
    
    if not train_csv.exists() or not test_csv.exists():
        print(f"\n✗ Error: CSV files not found!")
        print(f"  Expected: {train_csv}")
        print(f"  Expected: {test_csv}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Loading data splits...")
    print(f"{'='*60}")
    
    # Load pairs
    train_pairs = load_csv_pairs(train_csv)
    test_pairs = load_csv_pairs(test_csv)
    
    print(f"✓ Loaded {len(train_pairs)} training samples")
    print(f"✓ Loaded {len(test_pairs)} test samples")
    
    # Split train into train/val
    print(f"\nSplitting train into train/val ({1-args.val_ratio:.0%}/{args.val_ratio:.0%})...")
    train_pairs, val_pairs = split_train_val(train_pairs, args.val_ratio, args.seed)
    
    print(f"✓ Train: {len(train_pairs)} samples")
    print(f"✓ Val:   {len(val_pairs)} samples")
    print(f"✓ Test:  {len(test_pairs)} samples")
    
    # Create splits dictionary
    splits = {
        'train': [
            {
                'rgb': str(input_dir / pair['rgb']),
                'label': str(input_dir / pair['label'])
            }
            for pair in train_pairs
        ],
        'val': [
            {
                'rgb': str(input_dir / pair['rgb']),
                'label': str(input_dir / pair['label'])
            }
            for pair in val_pairs
        ],
        'test': [
            {
                'rgb': str(input_dir / pair['rgb']),
                'label': str(input_dir / pair['label'])
            }
            for pair in test_pairs
        ]
    }
    
    # Add metadata
    splits['metadata'] = {
        'dataset': 'NYUv2',
        'num_classes': 13,  # Semantic segmentation classes
        'tasks': ['segmentation', 'depth', 'normal'],
        'image_size': [480, 640],  # Original size (H, W)
        'train_size': len(train_pairs),
        'val_size': len(val_pairs),
        'test_size': len(test_pairs),
        'val_ratio': args.val_ratio,
        'random_seed': args.seed
    }
    
    # Save splits.json
    splits_file = output_dir / "splits.json"
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✓ Saved data splits to: {splits_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Preprocessing Summary")
    print(f"{'='*60}")
    print(f"Dataset: NYUv2")
    print(f"Tasks: Semantic Segmentation (13 classes)")
    print(f"       Depth Estimation")
    print(f"       Surface Normal Prediction")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_pairs):>5} samples")
    print(f"  Val:   {len(val_pairs):>5} samples")
    print(f"  Test:  {len(test_pairs):>5} samples")
    print(f"  Total: {len(train_pairs) + len(val_pairs) + len(test_pairs):>5} samples")
    print(f"\nOutput: {output_dir}/splits.json")
    print(f"\nNote: Images will be resized to 288x288 on-the-fly by DataLoader")
    print(f"{'='*60}")
    print("✓ Preprocessing complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

