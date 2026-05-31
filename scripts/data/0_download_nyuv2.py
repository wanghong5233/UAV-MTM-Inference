"""
Download NYUv2 dataset for multi-task learning.
Supports semantic segmentation, depth estimation, and surface normal prediction.
"""

import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import argparse
import sys


def download_file(url, dest_path):
    """Download file with progress bar."""
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            downloaded = count * block_size
            percent = min(int(downloaded * 100 / total_size), 100)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            
            # Progress bar
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r[{bar}] {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
        else:
            print(f"\rDownloading... {count * block_size / (1024*1024):.1f} MB", end="", flush=True)
    
    urllib.request.urlretrieve(url, dest_path, progress_hook)
    print()  # New line after progress bar


def extract_archive(archive_path, extract_to):
    """Extract zip or tar.gz archive."""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    print("Extraction complete!")


def download_from_google_drive(file_id, dest_path):
    """Download file from Google Drive using gdown."""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest_path), quiet=False)
        return True
    except ImportError:
        print("\n⚠ gdown not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest_path), quiet=False)
        return True
    except Exception as e:
        print(f"✗ Google Drive download failed: {e}")
        return False


def verify_file_size(file_path, min_size_mb=100):
    """Verify downloaded file is not HTML redirect."""
    if not file_path.exists():
        return False
    
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb < min_size_mb:
        print(f"\n⚠ Downloaded file is too small ({size_mb:.1f} MB)")
        print("  This might be an HTML redirect page, not the actual dataset.")
        return False
    return True


def download_nyuv2(data_dir, auto_download=False):
    """
    Download NYUv2 dataset (preprocessed version).
    
    This uses the preprocessed version from:
    https://github.com/xaphoon/pytorch-nyuv2
    
    Contains:
    - 1449 RGB-D images (640x480)
    - Semantic segmentation labels (13 classes)
    - Depth maps
    - Surface normal maps
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = raw_dir / "nyuv2_raw"  # canonical root for this dataset
    zip_path = raw_dir / "nyu_v2.zip"
    
    print("\n" + "="*60)
    print("NYUv2 Dataset Download")
    print("="*60)
    
    # Check if already downloaded and extracted
    if dataset_dir.exists():
        print(f"\n✓ Dataset already exists: {dataset_dir}")
        print("Dataset is ready to use!")
        return True
    
    # Check if zip exists and valid
    if zip_path.exists() and verify_file_size(zip_path, min_size_mb=100):
        print(f"\n✓ Found valid zip file: {zip_path}")
        file_size = zip_path.stat().st_size / (1024**3)
        print(f"  Size: {file_size:.2f} GB")
        print("\nExtracting...")
        try:
            extract_archive(str(zip_path), str(raw_dir))
            # Some archives create a generic 'data' directory; rename it for clarity
            legacy_data_dir = raw_dir / "data"
            if legacy_data_dir.exists() and not dataset_dir.exists():
                legacy_data_dir.rename(dataset_dir)
            print(f"✓ Dataset extracted to: {dataset_dir}")
            return True
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            zip_path.unlink()  # Delete corrupted file
            print("Deleted corrupted file. Please try downloading again.")
            return False
    
    # Need to download
    print("\nDataset not found. Choose download method:")
    print("1. Auto-download from Google Drive (~2.8 GB) [Recommended]")
    print("2. Manual download instructions")
    
    if auto_download:
        choice = "1"
    else:
        choice = input("\nEnter choice (1/2): ").strip()
    
    if choice == "1":
        # Try Google Drive first (most reliable)
        print(f"\n{'='*60}")
        print("Downloading from Google Drive...")
        print(f"{'='*60}")
        print(f"Destination: {zip_path}")
        print("This may take 10-30 minutes depending on your network...")
        print("(You can press Ctrl+C to cancel and try manual download)\n")
        
        try:
            # Google Drive file ID for NYUv2
            file_id = "1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw"
            
            if download_from_google_drive(file_id, zip_path):
                # Verify download
                if verify_file_size(zip_path, min_size_mb=100):
                    print("\n✓ Download complete!")
                    
                    # Extract
                    print("\nExtracting...")
                    extract_archive(str(zip_path), str(raw_dir))
                    # Normalize directory name
                    legacy_data_dir = raw_dir / "data"
                    if legacy_data_dir.exists() and not dataset_dir.exists():
                        legacy_data_dir.rename(dataset_dir)
                    print(f"✓ Dataset extracted to: {dataset_dir}")
                    return True
                else:
                    if zip_path.exists():
                        zip_path.unlink()
                    raise Exception("Downloaded file is invalid")
            else:
                raise Exception("Download failed")
                
        except KeyboardInterrupt:
            print("\n\n✗ Download cancelled by user")
            if zip_path.exists():
                zip_path.unlink()
            print("\nYou can try manual download (option 2)")
            return False
            
        except Exception as e:
            print(f"\n✗ Auto-download failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
            print("\n" + "="*60)
            print("Please use manual download method:")
            print("="*60)
            show_manual_instructions(zip_path)
            return False
    
    else:
        # Manual download instructions
        show_manual_instructions(zip_path)
        return False


def show_manual_instructions(zip_path):
    """Show manual download instructions."""
    print("\n" + "="*60)
    print("Manual Download Instructions")
    print("="*60)
    print("\n1. Visit one of these links in your browser:")
    print("\n   Option A (Recommended):")
    print("   https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw")
    print("   - Click 'Download' button")
    print("   - If Google says 'virus scan warning', click 'Download anyway'")
    print("\n   Option B (Backup):")
    print("   https://www.dropbox.com/s/kfp1ny9kh25rj3m/nyu_v2.zip")
    print("   - Click 'Download' button")
    print(f"\n2. Save the file as: {zip_path.absolute()}")
    print(f"\n3. Verify file size is ~2.8 GB (not a few KB)")
    print("\n4. Run this script again to extract and normalize to 'nyuv2_raw/':")
    print(f"   python scripts/data/0_download_nyuv2.py")
    
    print("\n" + "="*60)
    print("Dataset Info:")
    print("="*60)
    print("- Images: 1449 RGB-D pairs")
    print("- Resolution: 640x480 (will be resized to 288x288 for training)")
    print("- Tasks: Semantic Segmentation (13 classes)")
    print("         Depth Estimation (metric depth)")
    print("         Surface Normal Prediction (3-channel)")
    print("- Train/Val split: ~1200/249")


def download_pretrained_models(data_dir):
    """
    Download pretrained multi-task models (optional).
    These can be used as starting points for fine-tuning.
    """
    pretrained_dir = Path(data_dir) / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Pretrained Models (Optional)")
    print("="*60)
    print("\nTo download pretrained models, visit:")
    print("https://github.com/lorenmt/mtan/releases")
    print(f"\nSave checkpoints to: {pretrained_dir}")
    print("\nAvailable models:")
    print("- mtan_nyuv2.pth (MTAN pretrained on NYUv2)")
    print("- split_network_nyuv2.pth (Hard sharing baseline)")


def main():
    parser = argparse.ArgumentParser(description="Download NYUv2 dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically download from Google Drive (no interaction)"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Show pretrained model download instructions"
    )
    
    args = parser.parse_args()
    
    # Download dataset
    success = download_nyuv2(args.data_dir, auto_download=args.auto)
    
    # Show model download info
    if args.models:
        download_pretrained_models(args.data_dir)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
