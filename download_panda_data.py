#!/usr/bin/env python3
"""
Download PANDA dataset from Kaggle and extract patches
"""
import os
import sys
import json
import subprocess
from pathlib import Path

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    print("=" * 70)
    print("SETTING UP KAGGLE CREDENTIALS")
    print("=" * 70)
    
    # Check if kaggle.json exists in current directory
    if os.path.exists('kaggle_credentials.json'):
        print("✓ Found kaggle_credentials.json")
        
        # Create .kaggle directory
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        # Copy credentials
        with open('kaggle_credentials.json', 'r') as f:
            credentials = json.load(f)
        
        kaggle_json = kaggle_dir / 'kaggle.json'
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f)
        
        # Set permissions
        os.chmod(kaggle_json, 0o600)
        print(f"✓ Credentials saved to {kaggle_json}")
        print(f"✓ Permissions set to 600")
        return True
    else:
        print("✗ kaggle_credentials.json not found")
        print("\nPlease create kaggle_credentials.json with:")
        print('{')
        print('  "username": "your_kaggle_username",')
        print('  "key": "your_kaggle_api_key"')
        print('}')
        return False

def install_kaggle():
    """Install Kaggle API"""
    print("\n" + "=" * 70)
    print("INSTALLING KAGGLE API")
    print("=" * 70)
    
    try:
        import kaggle
        print("✓ Kaggle API already installed")
        return True
    except ImportError:
        print("Installing Kaggle API...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        print("✓ Kaggle API installed")
        return True

def download_dataset():
    """Download PANDA dataset"""
    print("\n" + "=" * 70)
    print("DOWNLOADING PANDA DATASET")
    print("=" * 70)
    
    data_dir = Path('./panda_data')
    data_dir.mkdir(exist_ok=True)
    
    print(f"Download directory: {data_dir.absolute()}")
    print("\nThis will download:")
    print("  - train.csv (~1 MB)")
    print("  - train_images/ (~100 GB of TIFF files)")
    print("\nNote: This is a large dataset and will take time!")
    
    response = input("\nProceed with download? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Download cancelled")
        return False
    
    print("\nDownloading dataset...")
    try:
        subprocess.run([
            'kaggle', 'competitions', 'download',
            '-c', 'prostate-cancer-grade-assessment',
            '-p', str(data_dir)
        ], check=True)
        
        print("✓ Dataset downloaded")
        
        # Unzip files
        print("\nExtracting files...")
        import zipfile
        
        for zip_file in data_dir.glob('*.zip'):
            print(f"Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"✓ Extracted {zip_file.name}")
        
        print("\n✓ All files extracted")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading dataset: {e}")
        return False

def verify_download():
    """Verify downloaded files"""
    print("\n" + "=" * 70)
    print("VERIFYING DOWNLOAD")
    print("=" * 70)
    
    data_dir = Path('./panda_data')
    
    # Check for train.csv
    train_csv = data_dir / 'train.csv'
    if train_csv.exists():
        print(f"✓ train.csv found")
        import pandas as pd
        df = pd.read_csv(train_csv)
        print(f"  Total images: {len(df)}")
        print(f"  Grade distribution:")
        for grade, count in df['isup_grade'].value_counts().sort_index().items():
            print(f"    Grade {grade}: {count}")
    else:
        print("✗ train.csv not found")
        return False
    
    # Check for train_images directory
    train_images = data_dir / 'train_images'
    if train_images.exists():
        tiff_files = list(train_images.glob('*.tiff'))
        print(f"✓ train_images/ found")
        print(f"  TIFF files: {len(tiff_files)}")
    else:
        print("✗ train_images/ not found")
        return False
    
    return True

def main():
    print("=" * 70)
    print("PANDA DATASET DOWNLOADER")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Setup Kaggle API credentials")
    print("2. Install Kaggle API (if needed)")
    print("3. Download PANDA dataset (~100 GB)")
    print("4. Extract files")
    print("5. Verify download")
    print("\n" + "=" * 70)
    
    # Step 1: Setup credentials
    if not setup_kaggle_credentials():
        print("\n✗ Failed to setup credentials")
        return 1
    
    # Step 2: Install Kaggle API
    if not install_kaggle():
        print("\n✗ Failed to install Kaggle API")
        return 1
    
    # Step 3: Download dataset
    if not download_dataset():
        print("\n✗ Failed to download dataset")
        return 1
    
    # Step 4: Verify
    if not verify_download():
        print("\n✗ Download verification failed")
        return 1
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Extract patches: python extract_patches.py")
    print("2. Verify data: python check_data.py")
    print("3. Start training: python train.py --epochs 200 --batch_size 32")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
