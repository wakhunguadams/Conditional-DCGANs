# Data Preparation Guide

## Overview

This guide will help you download the PANDA dataset and extract patches on your workstation.

## Prerequisites

- Kaggle account
- Kaggle API credentials
- ~100 GB free disk space
- Python 3.8+

## Your Kaggle Credentials

**Username:** `adamswakhungu`  
**API Key:** `9ebe23cf774a76bce5d1ca1bb384434c`

These credentials are stored in `kaggle_credentials.json` (excluded from Git for security).

## Quick Start

### Option 1: Automated Download and Extraction

```bash
# Activate environment
source venv/bin/activate

# Download dataset
python download_panda_data.py

# Extract patches
python extract_patches.py

# Verify
python check_data.py
```

### Option 2: Manual Steps

#### Step 1: Setup Kaggle API

```bash
# Install Kaggle API
pip install kaggle

# Create .kaggle directory
mkdir -p ~/.kaggle

# Copy credentials
cp kaggle_credentials.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

#### Step 2: Download Dataset

```bash
# Download PANDA dataset (~100 GB)
kaggle competitions download -c prostate-cancer-grade-assessment -p ./panda_data

# Extract files
cd panda_data
unzip train.csv.zip
unzip train_images.zip
cd ..
```

#### Step 3: Extract Patches

```bash
# Extract patches (default: 256x256, max 20 per image)
python extract_patches.py

# Or with custom settings
python extract_patches.py \
  --patch_size 256 \
  --max_patches 20 \
  --max_per_grade 500 \
  --tissue_thresh 0.5
```

#### Step 4: Verify Data

```bash
python check_data.py
```

## Extraction Options

### Basic Extraction
```bash
python extract_patches.py
```

### Custom Settings
```bash
# More patches per image
python extract_patches.py --max_patches 30

# Larger dataset
python extract_patches.py --max_per_grade 1000

# Different patch size
python extract_patches.py --patch_size 512

# Lower tissue threshold (more patches)
python extract_patches.py --tissue_thresh 0.3

# All options
python extract_patches.py \
  --patch_size 256 \
  --max_patches 30 \
  --max_per_grade 1000 \
  --tissue_thresh 0.4
```

## Expected Output

### After Download
```
panda_data/
├── train.csv                    # Labels (~1 MB)
└── train_images/                # TIFF files (~100 GB)
    ├── 0005f7aaab2800f6170c399693a96917.tiff
    ├── 000920ad0b612851f8e01bcc880d9b3d.tiff
    └── ... (10,616 images)
```

### After Extraction
```
panda_data/
├── train.csv
├── train_images/
└── patches_256/                 # Extracted patches
    ├── 0/                       # Grade 0 (Benign)
    ├── 1/                       # Grade 1 (G3+3)
    ├── 2/                       # Grade 2 (G3+4)
    ├── 3/                       # Grade 3 (G4+3)
    ├── 4/                       # Grade 4 (G4+4)
    └── 5/                       # Grade 5 (High)
```

## Dataset Statistics

### PANDA Dataset
- **Total images:** 10,616 whole slide images
- **Format:** TIFF (multi-resolution)
- **Size:** ~100 GB
- **Labels:** ISUP grades 0-5

### Grade Distribution (Full Dataset)
```
Grade 0 (Benign):     ~3,000 images
Grade 1 (G3+3):       ~2,000 images
Grade 2 (G3+4):       ~1,500 images
Grade 3 (G4+3):       ~1,500 images
Grade 4 (G4+4):       ~1,500 images
Grade 5 (High):       ~1,000 images
```

### Expected Patches (with defaults)
```
Max 20 patches per image × 500 images per grade = ~10,000 patches per grade
Total: ~60,000 patches (256×256 pixels)
```

## Troubleshooting

### Issue: Kaggle API Not Found
```bash
pip install kaggle
```

### Issue: Authentication Failed
```bash
# Check credentials
cat ~/.kaggle/kaggle.json

# Should show:
# {"username":"adamswakhungu","key":"9ebe23cf774a76bce5d1ca1bb384434c"}

# Fix permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: Download Interrupted
```bash
# Resume download
kaggle competitions download -c prostate-cancer-grade-assessment -p ./panda_data
```

### Issue: Not Enough Disk Space
```bash
# Check available space
df -h

# Extract fewer patches
python extract_patches.py --max_per_grade 100
```

### Issue: Extraction Too Slow
```bash
# Process fewer images per grade
python extract_patches.py --max_per_grade 200

# Or extract fewer patches per image
python extract_patches.py --max_patches 10
```

### Issue: Too Many Background Patches
```bash
# Increase tissue threshold
python extract_patches.py --tissue_thresh 0.7
```

## Verification

After extraction, verify your data:

```bash
python check_data.py
```

Expected output:
```
============================================================
PANDA Dataset Statistics
============================================================

Grade 0: Benign (no cancer)
  Patches: XXXX

Grade 1: Gleason 3+3
  Patches: XXXX

Grade 2: Gleason 3+4
  Patches: XXXX

Grade 3: Gleason 4+3
  Patches: XXXX

Grade 4: Gleason 4+4, 3+5, 5+3
  Patches: XXXX

Grade 5: Gleason 4+5, 5+4, 5+5
  Patches: XXXX

============================================================
Total patches: XXXXX
============================================================
```

## Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Download dataset | 1-3 hours | Depends on internet speed |
| Extract patches | 2-4 hours | Depends on CPU and settings |
| Verification | 1 minute | Quick check |

## Storage Requirements

| Component | Size |
|-----------|------|
| Downloaded ZIP files | ~100 GB |
| Extracted TIFF files | ~100 GB |
| Extracted patches (60K) | ~5-10 GB |
| **Total** | **~200-210 GB** |

**Tip:** You can delete the ZIP files after extraction to save space.

## Next Steps

After data preparation:

1. ✅ Verify data: `python check_data.py`
2. ✅ Start training: `python train.py --epochs 200 --batch_size 32`
3. ✅ Monitor progress: `tail -f logs/training.log`
4. ✅ Generate images: `python generate_by_grade.py --checkpoint checkpoints/G_final.pt`

## Security Note

⚠️ **IMPORTANT:** The `kaggle_credentials.json` file contains your API credentials and is excluded from Git via `.gitignore`. Never commit this file to a public repository!

If you need to share the project:
1. Remove `kaggle_credentials.json` before sharing
2. Share credentials separately via secure channel
3. Recipient should create their own `kaggle_credentials.json`

---

**Ready to download!** Run `python download_panda_data.py` to start. 🚀
