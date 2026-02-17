# Workstation Data Download Guide

## Overview
This guide explains how to download the PANDA dataset on your workstation, with options to get more data than your current setup.

## Current Setup Summary
- **Current images**: ~115 TIFF files
- **Current patches**: Extracted from grades 0-5
- **Patch size**: 256x256
- **Max patches per image**: 20

---

## Option 1: Quick Start - Download Same Data (Recommended First)

### Step 1: Transfer Kaggle Credentials
```bash
# On your local machine, copy kaggle_credentials.json to workstation
scp kaggle_credentials.json user@workstation:/path/to/project/
```

### Step 2: Run Download Script
```bash
# On workstation
cd /path/to/project
python download_panda_data.py
```

This will:
- Setup Kaggle API credentials
- Download the full PANDA dataset (~100 GB)
- Extract train.csv and train_images/

### Step 3: Extract Patches (Same as Current)
```bash
# Extract same amount as current setup
python extract_patches.py \
    --max_per_grade 500 \
    --max_patches 20 \
    --patch_size 256
```

---

## Option 2: Download MORE Data (Recommended for Workstation)

Since you have a bigger machine, you can extract more patches for better training:

### Increase Patches Per Image
```bash
# Extract 50 patches per image instead of 20
python extract_patches.py \
    --max_per_grade 1000 \
    --max_patches 50 \
    --patch_size 256
```

### Increase Images Per Grade
```bash
# Process 2000 images per grade instead of 500
python extract_patches.py \
    --max_per_grade 2000 \
    --max_patches 20 \
    --patch_size 256
```

### Maximum Data Extraction
```bash
# Extract ALL available data (will take hours)
python extract_patches.py \
    --max_per_grade 999999 \
    --max_patches 50 \
    --patch_size 256
```

---

## Option 3: Automated Workstation Setup Script

I'll create a script that does everything automatically:

```bash
# Run this on workstation
bash setup_workstation_data.sh
```

This script will:
1. Check for Kaggle credentials
2. Download full dataset
3. Extract patches with workstation-optimized settings
4. Verify data integrity
5. Show summary statistics

---

## Data Size Estimates

| Configuration | Images/Grade | Patches/Image | Total Patches | Disk Space |
|--------------|--------------|---------------|---------------|------------|
| Current (small) | 500 | 20 | ~60,000 | ~4 GB |
| Medium | 1000 | 30 | ~180,000 | ~12 GB |
| Large | 2000 | 50 | ~600,000 | ~40 GB |
| Maximum | All (~2000) | 50 | ~600,000 | ~40 GB |

---

## Step-by-Step Workstation Setup

### 1. Prepare Kaggle Credentials

**Option A: Copy from local machine**
```bash
scp kaggle_credentials.json user@workstation:/path/to/project/
```

**Option B: Create manually on workstation**
```bash
# On workstation
cat > kaggle_credentials.json << 'EOF'
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}
EOF
```

Get your API key from: https://www.kaggle.com/settings/account

### 2. Install Dependencies
```bash
# On workstation
cd /path/to/project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install kaggle
```

### 3. Download Dataset
```bash
python download_panda_data.py
```

**Expected output:**
```
======================================================================
PANDA DATASET DOWNLOADER
======================================================================
✓ Found kaggle_credentials.json
✓ Credentials saved to ~/.kaggle/kaggle.json
✓ Kaggle API already installed
Downloading dataset...
✓ Dataset downloaded
Extracting files...
✓ All files extracted
✓ train.csv found
  Total images: 10616
✓ train_images/ found
  TIFF files: 10616
```

### 4. Extract Patches (Choose Your Size)

**Small (like current):**
```bash
python extract_patches.py --max_per_grade 500 --max_patches 20
```

**Medium (recommended for workstation):**
```bash
python extract_patches.py --max_per_grade 1500 --max_patches 35
```

**Large (maximum data):**
```bash
python extract_patches.py --max_per_grade 999999 --max_patches 50
```

### 5. Verify Data
```bash
python check_data.py
```

---

## Advanced: Parallel Patch Extraction

For faster extraction on workstation with multiple cores:

```python
# I can create a parallel version of extract_patches.py
# that uses multiprocessing to speed up extraction
```

---

## Troubleshooting

### Issue: Kaggle API Not Authenticated
```bash
# Check credentials
cat ~/.kaggle/kaggle.json

# Re-setup credentials
python download_panda_data.py
```

### Issue: Out of Disk Space
```bash
# Check available space
df -h

# Reduce max_per_grade or max_patches
python extract_patches.py --max_per_grade 500 --max_patches 15
```

### Issue: Download Interrupted
```bash
# Resume download
kaggle competitions download -c prostate-cancer-grade-assessment -p ./panda_data
```

### Issue: Slow Extraction
```bash
# Use fewer patches per image
python extract_patches.py --max_patches 10

# Or process fewer images
python extract_patches.py --max_per_grade 300
```

---

## Recommended Workstation Configuration

Based on typical workstation specs:

**For 32GB RAM + 500GB Storage:**
```bash
python extract_patches.py \
    --max_per_grade 1500 \
    --max_patches 35 \
    --patch_size 256 \
    --tissue_thresh 0.5
```

**For 64GB RAM + 1TB Storage:**
```bash
python extract_patches.py \
    --max_per_grade 999999 \
    --max_patches 50 \
    --patch_size 256 \
    --tissue_thresh 0.5
```

---

## Next Steps After Data Download

1. **Verify data**: `python check_data.py`
2. **Transfer checkpoints** (if continuing training): `scp -r checkpoints/ user@workstation:/path/to/project/`
3. **Start training**: `python train_improved.py --epochs 200 --batch_size 64`
4. **Monitor with tmux**: See TMUX_GUIDE.md

---

## Quick Reference Commands

```bash
# Download full dataset
python download_panda_data.py

# Extract patches (medium size)
python extract_patches.py --max_per_grade 1500 --max_patches 35

# Verify data
python check_data.py

# Check disk usage
du -sh panda_data/

# Count patches
find panda_data/patches_256 -name "*.png" | wc -l
```
