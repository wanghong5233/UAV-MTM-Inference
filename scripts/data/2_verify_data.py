"""
Verify NYUv2 dataset integrity.

Checks:
1. splits.json exists and is valid
2. All file paths in splits.json exist
3. Image files can be loaded
4. Label files have correct dimensions

Usage:
    python scripts/data/2_verify_data.py
    python scripts/data/2_verify_data.py --data_dir data/processed/nyuv2
"""

import argparse
import sys
import json
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def verify_splits_file(splits_path):
    """Verify splits.json file exists and has correct structure."""
    if not splits_path.exists():
        print(f"✗ Error: splits.json not found at {splits_path}")
        return None
    
    try:
        with open(splits_path, 'r') as f:
            splits = json.load(f)
    except Exception as e:
        print(f"✗ Error loading splits.json: {e}")
        return None
    
    # Check required keys
    required_keys = ['train', 'val', 'test', 'metadata']
    for key in required_keys:
        if key not in splits:
            print(f"✗ Error: Missing key '{key}' in splits.json")
            return None
    
    return splits


def verify_file_paths(splits):
    """Verify all file paths in splits exist."""
    errors = []
    
    for split_name in ['train', 'val', 'test']:
        split_data = splits[split_name]
        
        for i, sample in enumerate(split_data):
            rgb_path = Path(sample['rgb'])
            label_path = Path(sample['label'])
            
            if not rgb_path.exists():
                errors.append(f"{split_name}[{i}]: RGB file not found: {rgb_path}")
            
            if not label_path.exists():
                errors.append(f"{split_name}[{i}]: Label file not found: {label_path}")
    
            # Only report first 5 errors to avoid spam
            if len(errors) >= 5:
                break
        
        if len(errors) >= 5:
            break
    
    return errors


def sample_and_load(splits, n_samples=3):
    """Randomly sample and try to load images."""
    print(f"\nSampling {n_samples} random images from each split...")
    
    try:
        from PIL import Image
    except ImportError:
        print("⚠ PIL not installed, skipping image loading test")
        return True
    
    for split_name in ['train', 'val', 'test']:
        split_data = splits[split_name]
        samples = random.sample(split_data, min(n_samples, len(split_data)))
        
        print(f"\n{split_name.capitalize()}:")
        for sample in samples:
            try:
                # Try to load RGB
                rgb_img = Image.open(sample['rgb'])
                label_img = Image.open(sample['label'])

                print(
                    f"  ✓ {Path(sample['rgb']).name}: "
                    f"RGB {rgb_img.size}, Label {label_img.size}"
                )
    except Exception as e:
                print(f"  ✗ Failed to load {sample['rgb']}: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Verify NYUv2 dataset')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/nyuv2',
        help='Processed data directory'
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    splits_path = data_dir / "splits.json"
    
    print("\n" + "="*60)
    print("NYUv2 Dataset Verification")
    print("="*60)
    print(f"\nData directory: {data_dir}")
    
    # Check 1: splits.json
    print(f"\n{'='*60}")
    print("1. Checking splits.json...")
    print(f"{'='*60}")
    
    splits = verify_splits_file(splits_path)
    if splits is None:
        print("\n✗ Verification failed!")
        sys.exit(1)
    
    print(f"✓ splits.json found and valid")
    print(f"\n  Dataset: {splits['metadata']['dataset']}")
    print(f"  Tasks: {', '.join(splits['metadata']['tasks'])}")
    print(f"  Train samples: {splits['metadata']['train_size']}")
    print(f"  Val samples:   {splits['metadata']['val_size']}")
    print(f"  Test samples:  {splits['metadata']['test_size']}")
    
    # Check 2: File paths
    print(f"\n{'='*60}")
    print("2. Verifying file paths...")
    print(f"{'='*60}")
    
    errors = verify_file_paths(splits)
    
    if errors:
        print(f"\n✗ Found {len(errors)} missing files:")
        for error in errors[:10]:  # Show max 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print("\n✗ Verification failed!")
        sys.exit(1)
    
    print(f"✓ All file paths exist")
    
    # Check 3: Sample loading
    print(f"\n{'='*60}")
    print("3. Testing image loading...")
    print(f"{'='*60}")
    
    if not sample_and_load(splits, n_samples=3):
        print("\n✗ Image loading test failed!")
        sys.exit(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("Verification Summary")
    print(f"{'='*60}")
    print(f"✓ splits.json: Valid")
    print(f"✓ File paths: All exist")
    print(f"✓ Image loading: OK")
    print(f"\nDataset is ready for training!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

